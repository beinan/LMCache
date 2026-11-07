# SPDX-License-Identifier: Apache-2.0
"""Request-owned CacheBlend lookup contracts, with CPU tensors and fake RPCs."""

# Standard
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
import threading
import unittest

# Third Party
import torch

# First Party
from lmcache.integration.sglang.lmcache_mp_metadata import (
    LMCacheLoadOperation,
    SGLangKVComponentGroup,
)
from lmcache.integration.sglang.unified_lmcache_mp_connector import (
    UnifiedLMCacheMPConnector,
)
from lmcache.v1.multiprocess.custom_types import CBMatchResult, CBUnifiedLookupResult
from lmcache.v1.multiprocess.futures import MessagingFuture


class TestSGLangBlendLookup(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()
        self.client.get_chunk_size.return_value.result.return_value = 8
        self.client.cb_protocol_handshake.return_value.result.return_value = (1, True)
        self.client.get_experimental.return_value.result.return_value = [
            "blend_rotary_components"
        ]
        self.future: MessagingFuture[CBUnifiedLookupResult | None] = MessagingFuture()
        self.client.cb_unified_lookup.return_value = self.future
        self.release: MessagingFuture[None] = MessagingFuture()
        self.release.set_result(None)
        self.client.free_lookup_locks.return_value = self.release
        self.ended: MessagingFuture[None] = MessagingFuture()
        self.ended.set_result(None)
        self.client.end_session.return_value = self.ended
        self.transfer = Mock()
        self.connector = self._make_connector(
            [
                SGLangKVComponentGroup(
                    "full",
                    tuple(torch.empty(32, 2, 8) for _ in range(2)),
                    rope_num_layers=1,
                )
            ]
        )

    def test_sparse_only_hit_survives_pending_polls_and_finishes_once(self) -> None:
        result = CBUnifiedLookupResult(0, [CBMatchResult(0, 8, 8, 16, b"hash")])
        self.future.set_result(None)
        complete: MessagingFuture[CBUnifiedLookupResult | None] = MessagingFuture()
        complete.set_result(result)
        self.client.cb_unified_lookup.side_effect = [self.future, complete]
        operation = self.connector.submit_blend_lookup(
            "sparse", list(range(16)), local_hit_tokens=0, cache_salt="tenant"
        )
        self.assertIsNone(self.connector.poll_blend_lookup(operation))
        self.assertIsNone(self.connector.poll_blend_lookup(operation))
        self.assertIs(self.connector.poll_blend_lookup(operation), result)
        self.assertEqual(operation.total_hit_tokens, 0)
        self.assertEqual(self.connector.poll_lookup(operation), 0)
        self.assertIs(self.connector.poll_blend_lookup(operation), result)
        self.assertEqual(self.client.cb_unified_lookup.call_count, 2)
        key = self.client.cb_unified_lookup.call_args.args[0]
        self.assertEqual(key.cache_salt, "tenant")
        self.connector.finish_request("sparse")
        self.connector.finish_request("sparse")
        self.client.end_session.assert_called_once_with(
            self.client.cb_unified_lookup.call_args.args[0].request_id
        )
        with self.assertRaisesRegex(RuntimeError, "retired"):
            self.connector.poll_blend_lookup(operation)

    def test_releasing_prefix_keeps_sparse_and_segmented_matches(self) -> None:
        result = CBUnifiedLookupResult(
            8,
            [CBMatchResult(0, 8, 24, 32, b"shifted")],
            [CBMatchResult(16, 24, 16, 24, b"exact")],
        )
        self.future.set_result(result)
        operation = self.connector.submit_blend_lookup(
            "mixed", list(range(32)), local_hit_tokens=8, cache_salt=""
        )
        self.assertIs(self.connector.poll_blend_lookup(operation), result)
        self.connector.free_lookup_locks("mixed", 0, 8)
        self.assertIs(self.connector.poll_blend_lookup(operation), result)
        self.assertEqual(operation.total_hit_tokens, 8)
        self.connector.finish_request("mixed")
        self.client.end_session.assert_called_once_with(
            self.client.cb_unified_lookup.call_args.args[0].request_id
        )

    def test_cancellation_waits_for_lookup_reply_without_resubmitting(self) -> None:
        operation = self.connector.submit_blend_lookup(
            "cancelled", list(range(16)), local_hit_tokens=0, cache_salt=""
        )
        self.assertIsNone(self.connector.poll_blend_lookup(operation))
        self.connector.finish_request("cancelled")
        self.connector.finish_request("cancelled")
        try:
            self.client.end_session.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "retired"):
                self.connector.poll_blend_lookup(operation)
        finally:
            self.future.set_result(None)
        self.client.cb_unified_lookup.assert_called_once()
        self.client.end_session.assert_called_once_with(
            self.client.cb_unified_lookup.call_args.args[0].request_id
        )
        self.client.free_lookup_locks.assert_not_called()

    def test_cancellation_releases_unpolled_prefix_before_ending(self) -> None:
        self.release = MessagingFuture()
        self.client.free_lookup_locks.return_value = self.release
        self.connector.submit_blend_lookup(
            "unpolled", list(range(16)), local_hit_tokens=0, cache_salt="tenant"
        )
        self.connector.finish_request("unpolled")
        self.future.set_result(CBUnifiedLookupResult(16, []))
        try:
            self.client.free_lookup_locks.assert_called_once()
            key, readers = self.client.free_lookup_locks.call_args.args
            self.assertEqual(
                (key.start, key.end, key.cache_salt, readers), (0, 16, "tenant", 1)
            )
            self.client.end_session.assert_not_called()
        finally:
            self.release.set_result(None)
        self.client.end_session.assert_called_once_with(
            self.client.cb_unified_lookup.call_args.args[0].request_id
        )

    def test_reused_id_isolates_old_cleanup_from_new_transfers(self) -> None:
        old_end: MessagingFuture[None] = MessagingFuture()
        self.client.end_session.side_effect = [old_end, self.ended]
        self.connector.submit_blend_lookup(
            "reuse", list(range(16)), local_hit_tokens=0, cache_salt="tenant"
        )
        old_key = self.client.cb_unified_lookup.call_args.args[0]
        self.connector.finish_request("reuse")
        new_reply: MessagingFuture[CBUnifiedLookupResult | None] = MessagingFuture()
        new_reply.set_result(CBUnifiedLookupResult(8, []))
        self.client.cb_unified_lookup.return_value = new_reply
        new_lookup = self.connector.submit_blend_lookup(
            "reuse", list(range(16, 32)), local_hit_tokens=0, cache_salt="tenant"
        )
        new_key = self.client.cb_unified_lookup.call_args.args[0]
        try:
            self.assertNotEqual(old_key.request_id, new_key.request_id)
            self.assertEqual(new_lookup.request_id, "reuse")
            self.future.set_result(CBUnifiedLookupResult(16, []))
            released_key = self.client.free_lookup_locks.call_args.args[0]
            self.assertEqual(released_key.request_id, old_key.request_id)
            self.client.end_session.assert_called_once_with(old_key.request_id)
            self.assertEqual(self.connector.poll_lookup(new_lookup), 8)

            transferred: MessagingFuture[bool] = MessagingFuture()
            transferred.set_result(True)
            self.transfer.submit_retrieve.return_value = transferred
            loaded = self.connector.submit_load(
                new_lookup, torch.arange(8, 16), local_hit_tokens=0
            )
            load_args = self.transfer.submit_retrieve.call_args.args
            self.assertEqual(
                (load_args[0], load_args[1].request_id), ("reuse", new_key.request_id)
            )
            self.assertTrue(self.connector.complete_load(loaded))

            self.transfer.submit_store.return_value = transferred
            stored = self.connector.submit_store(
                "reuse",
                list(range(16, 32)),
                torch.arange(8, 24),
                device_indices_start=0,
                cache_salt="tenant",
            )
            assert stored is not None
            self.assertEqual(stored.request_id, "reuse")
            self.assertTrue(self.connector.complete_store(stored))
            store_key = self.transfer.submit_store.call_args.args[1]
            self.assertEqual(store_key.request_id, new_key.request_id)
            old_end.set_result(None)
            self.assertEqual(self.connector.poll_lookup(new_lookup), 8)
            self.connector.finish_request("reuse")
            self.client.end_session.assert_called_with(new_key.request_id)
            self.assertEqual(self.client.end_session.call_count, 2)
        finally:
            if not self.future.query():
                self.future.set_result(None)
            if not old_end.query():
                old_end.set_result(None)

    def test_short_blend_request_keeps_its_session_for_later_store(self) -> None:
        old_end: MessagingFuture[None] = MessagingFuture()
        self.client.end_session.side_effect = [old_end, self.ended]
        transferred: MessagingFuture[bool] = MessagingFuture()
        transferred.set_result(True)
        self.transfer.submit_store.return_value = transferred
        sessions: list[str] = []
        try:
            for offset in (0, 8):
                lookup = self.connector.submit_blend_lookup(
                    "growing", [offset], local_hit_tokens=0, cache_salt="tenant"
                )
                self.assertEqual(self.connector.poll_lookup(lookup), 0)
                stored = self.connector.submit_store(
                    "growing",
                    list(range(offset, offset + 8)),
                    torch.arange(8, 16),
                    device_indices_start=0,
                    cache_salt="tenant",
                )
                assert stored is not None
                self.assertTrue(self.connector.complete_store(stored))
                store_key = self.transfer.submit_store.call_args.args[1]
                sessions.append(store_key.request_id)
                self.connector.finish_request("growing")
                self.client.end_session.assert_called_with(store_key.request_id)
            self.client.cb_unified_lookup.assert_not_called()
            self.assertNotEqual(*sessions)
        finally:
            old_end.set_result(None)

    def test_blend_lookup_cannot_replace_a_live_store_session(self) -> None:
        transferred: MessagingFuture[bool] = MessagingFuture()
        transferred.set_result(True)
        self.transfer.submit_store.return_value = transferred
        stored = self.connector.submit_store(
            "stored",
            list(range(8)),
            torch.arange(8, 16),
            device_indices_start=0,
            cache_salt="tenant",
        )
        assert stored is not None
        self.assertTrue(self.connector.complete_store(stored))
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self.connector.submit_blend_lookup(
                "stored", list(range(8)), local_hit_tokens=0, cache_salt="tenant"
            )
        self.connector.finish_request("stored")
        self.client.end_session.assert_called_once_with("stored")

    def test_cancellation_waits_for_partial_release_and_frees_remainder(self) -> None:
        partial: MessagingFuture[None] = MessagingFuture()
        self.client.free_lookup_locks.side_effect = [partial, self.release]
        self.future.set_result(CBUnifiedLookupResult(16, []))
        operation = self.connector.submit_blend_lookup(
            "partial", list(range(16)), local_hit_tokens=0, cache_salt=""
        )
        self.connector.poll_blend_lookup(operation)
        self.connector.free_lookup_locks("partial", 0, 8)
        self.connector.finish_request("partial")
        try:
            self.client.end_session.assert_not_called()
        finally:
            partial.set_result(None)
        ranges = [
            (call.args[0].start, call.args[0].end)
            for call in self.client.free_lookup_locks.call_args_list
        ]
        self.assertEqual(ranges, [(0, 8), (8, 16)])
        self.client.end_session.assert_called_once_with(
            self.client.cb_unified_lookup.call_args.args[0].request_id
        )

    def test_finished_load_does_not_release_prefix_twice(self) -> None:
        self.future.set_result(CBUnifiedLookupResult(16, []))
        lookup = self.connector.submit_blend_lookup(
            "loaded", list(range(16)), local_hit_tokens=0, cache_salt=""
        )
        self.connector.poll_blend_lookup(lookup)
        retrieved: MessagingFuture[bool] = MessagingFuture()
        retrieved.set_result(True)
        load = LMCacheLoadOperation(
            "loaded", list(range(16)), 0, 16, 0, torch.arange(16), retrieved, lookup
        )
        self.assertTrue(self.connector.complete_load(load))
        self.connector.finish_request("loaded")
        self.client.free_lookup_locks.assert_not_called()
        self.client.end_session.assert_called_once_with(
            self.client.cb_unified_lookup.call_args.args[0].request_id
        )

    def test_lookup_error_still_ends_session(self) -> None:
        self.connector.submit_blend_lookup(
            "failed", list(range(16)), local_hit_tokens=0, cache_salt=""
        )
        self.connector.finish_request("failed")
        self.future.set_exception(ConnectionError("lookup failed"))
        self.client.end_session.assert_called_once_with(
            self.client.cb_unified_lookup.call_args.args[0].request_id
        )
        self.client.free_lookup_locks.assert_not_called()

    def test_close_waits_for_session_end_acknowledgement(self) -> None:
        end_submitted = threading.Event()
        ended: MessagingFuture[None] = MessagingFuture()

        def end_session(request_id: str) -> MessagingFuture[None]:
            end_submitted.set()
            return ended

        self.client.end_session.side_effect = end_session
        release: MessagingFuture[None] = MessagingFuture()
        self.client.free_lookup_locks.return_value = release
        self.connector.submit_blend_lookup(
            "closing", list(range(16)), local_hit_tokens=0, cache_salt=""
        )
        self.connector.finish_request("closing")
        closing = threading.Thread(target=self.connector.close)
        closing.start()
        try:
            self.client.end_session.assert_not_called()
            self.client.close.assert_not_called()
            self.future.set_result(CBUnifiedLookupResult(8, []))
            self.client.free_lookup_locks.assert_called_once()
            self.client.end_session.assert_not_called()
            self.client.close.assert_not_called()
            release.set_result(None)
            self.assertTrue(end_submitted.wait(5))
            self.client.close.assert_not_called()
        finally:
            if not self.future.query():
                self.future.set_result(None)
            if not release.query():
                release.set_result(None)
            ended.set_result(None)
            closing.join(5)
        self.assertFalse(closing.is_alive())
        self.client.close.assert_called_once()

    def test_completed_prefix_is_released_with_or_without_polling(self) -> None:
        for poll in (False, True):
            with self.subTest(poll=poll):
                future: MessagingFuture[CBUnifiedLookupResult | None] = (
                    MessagingFuture()
                )
                future.set_result(CBUnifiedLookupResult(16, []))
                self.client.cb_unified_lookup.return_value = future
                self.client.free_lookup_locks.reset_mock()
                self.client.end_session.reset_mock()
                request_id = f"completed-{poll}"
                operation = self.connector.submit_blend_lookup(
                    request_id, list(range(16)), local_hit_tokens=0, cache_salt=""
                )
                if poll:
                    self.connector.poll_blend_lookup(operation)
                self.connector.finish_request(request_id)
                self.client.free_lookup_locks.assert_called_once()
                key = self.client.free_lookup_locks.call_args.args[0]
                self.assertEqual((key.start, key.end), (0, 16))
                self.client.end_session.assert_called_once_with(
                    self.client.cb_unified_lookup.call_args.args[0].request_id
                )

    def test_release_error_still_ends_session(self) -> None:
        self.release = MessagingFuture()
        self.client.free_lookup_locks.return_value = self.release
        self.future.set_result(CBUnifiedLookupResult(8, []))
        operation = self.connector.submit_blend_lookup(
            "release-error", list(range(16)), local_hit_tokens=0, cache_salt=""
        )
        self.connector.poll_blend_lookup(operation)
        self.connector.finish_request("release-error")
        self.client.end_session.assert_not_called()
        self.release.set_exception(ConnectionError("release failed"))
        self.client.end_session.assert_called_once_with(
            self.client.cb_unified_lookup.call_args.args[0].request_id
        )

    def test_server_without_rotary_capability_rejected_before_lookup(self) -> None:
        self.future.set_result(CBUnifiedLookupResult(0, []))
        self.client.get_experimental.return_value.result.return_value = []
        for version in (1, 2):
            with self.subTest(version=version):
                self.client.cb_protocol_handshake.return_value.result.return_value = (
                    version,
                    True,
                )
                with self.assertRaisesRegex(RuntimeError, "blend_rotary_components"):
                    self.connector.submit_blend_lookup(
                        "old", list(range(16)), local_hit_tokens=0, cache_salt=""
                    )
                self.client.cb_unified_lookup.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, "blend_rotary_components"):
                    self.connector.register_blend_rope(None, is_neox_style=False)
                self.transfer.register_blend_rope.assert_not_called()
                self.connector.finish_request("old")
                self.client.end_session.assert_not_called()

    def test_newer_protocol_with_rotary_capability_is_accepted(self) -> None:
        self.client.cb_protocol_handshake.return_value.result.return_value = (2, True)
        result = CBUnifiedLookupResult(0, [])
        self.future.set_result(result)
        operation = self.connector.submit_blend_lookup(
            "newer", list(range(16)), local_hit_tokens=0, cache_salt=""
        )
        self.assertIs(self.connector.poll_blend_lookup(operation), result)
        self.connector.finish_request("newer")

    def test_incompatible_protocol_rejected_before_feature_query(self) -> None:
        self.client.cb_protocol_handshake.return_value.result.return_value = (1, False)
        with self.assertRaisesRegex(RuntimeError, "protocol"):
            self.connector.submit_blend_lookup(
                "incompatible", list(range(16)), local_hit_tokens=0, cache_salt=""
            )
        self.client.get_experimental.assert_not_called()
        self.client.cb_unified_lookup.assert_not_called()

    def test_short_prompt_never_creates_server_session(self) -> None:
        operation = self.connector.submit_blend_lookup(
            "short", [1], local_hit_tokens=0, cache_salt=""
        )
        self.assertEqual(
            self.connector.poll_blend_lookup(operation), CBUnifiedLookupResult(0, [])
        )
        self.connector.finish_request("short")
        self.client.cb_unified_lookup.assert_not_called()
        self.client.end_session.assert_not_called()

    def test_sparse_retrieve_uses_engine_group_tables_and_preserves_prefix(
        self,
    ) -> None:
        connector = self._make_connector(
            [
                SGLangKVComponentGroup(
                    f"full-{i}",
                    tuple(torch.empty(128, heads, head) for heads in head_counts),
                    rope_num_layers=len(head_counts) // 2,
                )
                for i, (head, head_counts) in enumerate(
                    ((8, (2, 4, 2, 4)), (16, (2, 2)))
                )
            ]
        )
        registered = self.transfer.register.call_args.kwargs["engine_group_infos"]
        self.assertEqual([group.engine_group_id for group in registered], [0, 0, 1])
        table = torch.ones(64, 4)
        connector.register_blend_rope(table, is_neox_style=True)
        args = self.transfer.register_blend_rope.call_args.args
        self.assertIs(args[0][0], table)
        self.assertEqual(args[1:6], (8, True, [0, 0], [[0, 4], [0, 4]], [8, 16]))
        sparse = CBMatchResult(0, 8, 24, 32, b"sparse")
        exact = CBMatchResult(16, 24, 16, 24, b"exact")
        self.future.set_result(CBUnifiedLookupResult(8, [sparse], [exact]))
        lookup = connector.submit_blend_lookup(
            "retrieve", list(range(32)), local_hit_tokens=0, cache_salt="tenant"
        )
        connector.poll_blend_lookup(lookup)
        completed: MessagingFuture[bool] = MessagingFuture()
        completed.set_result(True)
        self.transfer.submit_blend_retrieve.return_value = completed
        result = connector.submit_blend_retrieve(
            lookup, [torch.arange(32, 64), torch.arange(64, 96)]
        )
        self.assertIs(result, completed)
        key, matches, blocks, event = self.transfer.submit_blend_retrieve.call_args.args
        self.assertEqual(
            key.request_id, self.client.cb_unified_lookup.call_args.args[0].request_id
        )
        self.assertEqual(
            (key.start, key.end, key.worker_id, key.cache_salt), (0, 32, 0, "tenant")
        )
        self.assertEqual(matches, [sparse, exact])
        self.assertEqual(blocks, [list(range(8, 16)), list(range(16, 24))])
        self.assertIs(event, self.transfer.create_recorded_event.return_value)
        self.assertTrue(result.result(timeout=0))
        connector.finish_request("retrieve")
        self.client.free_lookup_locks.assert_called_once()
        key = self.client.free_lookup_locks.call_args.args[0]
        self.assertEqual((key.start, key.end), (0, 8))

    def test_sparse_retrieve_requires_owned_matches_and_complete_allocation(
        self,
    ) -> None:
        match = CBMatchResult(0, 8, 8, 16, b"sparse")
        self.future.set_result(CBUnifiedLookupResult(0, [match]))
        lookup = self.connector.submit_blend_lookup(
            "subset", list(range(16)), local_hit_tokens=0, cache_salt=""
        )
        with self.assertRaisesRegex(RuntimeError, "completed lookup"):
            self.connector.submit_blend_retrieve(lookup, torch.arange(16))
        self.connector.poll_blend_lookup(lookup)
        with self.assertRaisesRegex(RuntimeError, "register_blend_rope"):
            self.connector.submit_blend_retrieve(lookup, torch.arange(16))
        self.connector.register_blend_rope(None, is_neox_style=True)
        with self.assertRaisesRegex(ValueError, "cover every selected match"):
            self.connector.submit_blend_retrieve(lookup, torch.arange(8))
        for slots in (
            torch.cat((torch.arange(8, 16), torch.arange(8))),
            torch.arange(32, 48),
        ):
            with (
                self.subTest(slots=slots),
                self.assertRaisesRegex(ValueError, "non-padding"),
            ):
                self.connector.submit_blend_retrieve(lookup, slots)
        for matches in ([match, match], [CBMatchResult(0, 8, 8, 16, b"foreign")]):
            with (
                self.subTest(matches=matches),
                self.assertRaisesRegex(ValueError, "distinct matches"),
            ):
                self.connector.submit_blend_retrieve(
                    lookup, torch.arange(16), matches=matches
                )
        self.assertTrue(
            self.connector.submit_blend_retrieve(
                lookup, torch.arange(16), matches=[]
            ).result(0)
        )
        self.transfer.submit_blend_retrieve.assert_not_called()
        self.connector.finish_request("subset")
        with self.assertRaisesRegex(RuntimeError, "retired"):
            self.connector.submit_blend_retrieve(lookup, torch.arange(16))

    def test_rope_registration_rejects_invalid_geometry_and_failed_ack(self) -> None:
        for shape in ((8,), (16, 0), (16, 3), (16, 10)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.connector.register_blend_rope(
                    torch.empty(shape), is_neox_style=True
                )
        self.transfer.register_blend_rope.assert_not_called()
        self.transfer.register_blend_rope.side_effect = ConnectionError(
            "registration failed"
        )
        with self.assertRaises(ConnectionError):
            self.connector.register_blend_rope(None, is_neox_style=True)
        self.future.set_result(
            CBUnifiedLookupResult(0, [CBMatchResult(0, 8, 8, 16, b"h")])
        )
        lookup = self.connector.submit_blend_lookup(
            "failed-rope", list(range(16)), local_hit_tokens=0, cache_salt=""
        )
        self.connector.poll_blend_lookup(lookup)
        with self.assertRaisesRegex(RuntimeError, "register_blend_rope"):
            self.connector.submit_blend_retrieve(lookup, torch.arange(16))

    def _make_connector(
        self, groups: list[SGLangKVComponentGroup]
    ) -> UnifiedLMCacheMPConnector:
        adapter = Mock()
        adapter.resolve_registered_groups.return_value = groups
        adapter.is_mla_enabled.return_value = False
        module_name = "lmcache.integration.sglang.unified_kv_adapter"
        adapter_module = ModuleType(module_name)
        vars(adapter_module)["SGLangUnifiedKVAdapter"] = Mock(return_value=adapter)
        config = SimpleNamespace(mp_host="127.0.0.1", mp_port=5555)
        config.get_extra_config_value = lambda _key, default: default
        with (
            patch.dict("sys.modules", {module_name: adapter_module}),
            patch(
                "lmcache.v1.client.client_config.load_mp_client_config",
                return_value=config,
            ),
            patch(
                "lmcache.v1.multiprocess.transport.factory.RequestClientFactory.create",
                return_value=self.client,
            ),
            patch(
                "lmcache.v1.client.worker_transfer.create_transfer_context",
                return_value=self.transfer,
            ),
        ):
            connector = UnifiedLMCacheMPConnector(
                config_file=None,
                model_config=SimpleNamespace(model_path="test-model"),
                tp_size=1,
                tp_rank=0,
                tp_group=None,
                page_size=4,
                token_to_kv_pool_allocator=object(),
                req_to_token_pool=object(),
                tree_components=(object(),),
                mamba_component=None,
                sliding_window_size=None,
            )
        self.addCleanup(connector.close)
        return connector
