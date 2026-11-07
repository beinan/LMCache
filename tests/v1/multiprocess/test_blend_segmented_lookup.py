# SPDX-License-Identifier: Apache-2.0
"""Segmented lookup ownership through real storage and session interfaces."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    DEFAULT_SHARD_REPLICATION_DESC,
    AttnWindowDesc,
    GroupKind,
    MemoryLayoutDesc,
    PrefetchHandle,
    PrefetchResult,
    PrefetchTaskSpec,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import DramL1Manager
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import L1ManagerDescriptor
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.blend.matcher import BlendTokenRangeMatcher
from lmcache.v1.multiprocess.modules.blend.module import BlendModule
from lmcache.v1.multiprocess.modules.blend.store import fingerprint_namespace
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher


@pytest.mark.parametrize(
    "mode",
    [
        "plain",
        "overlap",
        "missing_aux",
        "recurrent",
        "cancel_prefix",
        "cancel_sparse",
        "repeat",
    ],
)
def test_segmented_lookup_owns_tail_until_complete_or_cancelled(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Only complete, disjoint matches survive; every tail lock has one owner."""
    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            init_size_in_bytes=1 << 20,
            use_lazy=False,
            mlock=False,
        )
    )
    l1 = DramL1Manager(cfg)
    ctrl = PrefetchController(
        [l1], [L1ManagerDescriptor(0, cfg)], [], [], DefaultPrefetchPolicy()
    )
    ctrl.start()
    ctx = MagicMock()
    ctx.chunk_size = 4
    ctx.token_hasher = TokenHasher(chunk_size=4)
    ctx.session_manager = SessionManager(ctx.token_hasher, cleanup_interval=None)
    ctx.event_bus.has_subscribers.return_value = False
    layout = MemoryLayoutDesc([torch.Size([4, 2])], [torch.bfloat16])
    extra = mode in ("overlap", "missing_aux", "recurrent")
    gids = [0, 1] if extra else [0]
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {
        g: layout for g in gids
    }
    kinds: tuple[GroupKind, ...] = (
        ("attention", "recurrent" if mode == "recurrent" else "aux")
        if extra
        else ("attention",)
    )
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1] * len(gids),
        world_size=1,
        group_kinds=kinds,
    )
    ctx.layout_desc_registry.find_shard_replication_desc.return_value = (
        DEFAULT_SHARD_REPLICATION_DESC
    )
    key = IPCCacheServerKey.from_token_ids(
        "model", 1, None, list(range(12)), request_id="query", end=12
    )
    hashes = ctx.token_hasher.compute_packed_chunk_hashes(key.token_bytes)
    grouped = ipc_key_to_object_keys(key, hashes, gids, DEFAULT_SHARD_REPLICATION_DESC)
    resident = [grouped[0][0], grouped[0][2]]
    prefix_keys = [grouped[0][0]]
    if extra and mode != "missing_aux":
        resident += [grouped[1][0], grouped[1][2]]
        if mode == "recurrent":
            prefix_keys.append(grouped[1][0])
    matcher = BlendTokenRangeMatcher(chunk_size=4)
    if mode in ("overlap", "missing_aux"):
        source = [7, 8, 9, 10]
        source_hashes = ctx.token_hasher.compute_chunk_hashes(source)
        matcher.on_new_token_hashes(
            source, source_hashes, namespace=fingerprint_namespace(key)
        )
        source_groups = ipc_key_to_object_keys(
            key, source_hashes, gids, DEFAULT_SHARD_REPLICATION_DESC
        )
        resident += [row[0] for row in source_groups]
    monkeypatch.setattr(
        "lmcache.v1.multiprocess.modules.blend.module.BlendTokenRangeMatcher",
        MagicMock(return_value=matcher),
    )
    written = l1.reserve_write(resident, [False] * len(resident), layout)
    assert all(written[k][0] == L1Error.SUCCESS for k in resident)
    l1.finish_write(resident)
    other_reader = l1.reserve_read(resident)
    assert all(other_reader[k][0] == L1Error.SUCCESS for k in resident)
    policies: dict[int, str] = {}

    def submit(spec: PrefetchTaskSpec, external_request_id: str) -> PrefetchHandle:
        handle = ctrl.submit_prefetch_request(spec)
        policies[handle] = spec.fetching_policy
        return PrefetchHandle(handle, external_request_id, 0, 0)

    def query(handle: PrefetchHandle) -> PrefetchResult | None:
        policy = policies[handle.prefetch_request_id]
        if (mode == "cancel_prefix" and policy == "segmented_prefix") or (
            mode == "cancel_sparse" and policy == "full"
        ):
            return None
        return ctrl.query_prefetch_result(handle.prefetch_request_id)

    ctx.storage_manager.submit_prefetch_task.side_effect = submit
    ctx.storage_manager.query_prefetch_status.side_effect = query
    ctx.storage_manager.discard_prefetch_task.side_effect = (
        lambda h: ctrl.discard_prefetch_result(h.prefetch_request_id)
    )
    ctx.storage_manager.finish_read_prefetched.side_effect = (
        lambda ks, read_locks=1, l1_owners=None: l1.finish_read(ks, read_locks)
    )
    blend = BlendModule(ctx, MagicMock(), enable_segmented_prefix=True)
    try:
        result = blend.cb_unified_lookup(key, 1)
        if mode.startswith("cancel_"):
            assert result is None
            assert l1.unsafe_read([grouped[0][2]])[grouped[0][2]][0] == L1Error.SUCCESS
        else:
            assert result is not None
            assert result.prefix_coverage_tokens == 4
            assert [(m.cur_st, m.cur_ed) for m in result.segmented_prefix_segments] == (
                [] if mode in ("missing_aux", "recurrent") else [(8, 12)]
            )
            assert [(m.cur_st, m.cur_ed) for m in result.non_prefix_segments] == (
                [(7, 11)] if mode == "missing_aux" else []
            )
            l1.finish_read(prefix_keys)
            if mode != "recurrent":
                ends = [
                    call.args[0]
                    for call in ctx.event_bus.publish.call_args_list
                    if call.args[0].event_type == EventType.CB_LOOKUP_END
                ]
                assert ends[-1].metadata["stale_chunks"] == 0
        if mode == "repeat":
            repeated = blend.cb_unified_lookup(key, 1)
            assert repeated == result
            l1.finish_read(prefix_keys)
        ctx.session_manager.remove(key.request_id)
        assert blend.report_status()["active_cb_lookups"] == 0
        assert all(
            error == L1Error.KEY_IS_LOCKED for error in l1.delete(resident).values()
        )
        l1.finish_read(resident)
        assert all(error == L1Error.SUCCESS for error in l1.delete(resident).values())
    finally:
        ctx.session_manager.close()
        blend.close()
        ctrl.stop()
        l1.close()
