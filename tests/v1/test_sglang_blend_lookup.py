# SPDX-License-Identifier: Apache-2.0
"""Request-owned CacheBlend lookup contracts, with CPU tensors and fake RPCs."""

# Standard
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
import unittest

# Third Party
import torch

# First Party
from lmcache.integration.sglang.lmcache_mp_metadata import SGLangKVComponentGroup
from lmcache.integration.sglang.unified_lmcache_mp_connector import (
    UnifiedLMCacheMPConnector,
)
from lmcache.v1.multiprocess.custom_types import CBMatchResult, CBUnifiedLookupResult


class TestSGLangBlendLookup(unittest.TestCase):
    def setUp(self) -> None:
        tensors = tuple(torch.empty(32, 2, 8) for _ in range(2))
        adapter = Mock()
        adapter.resolve_registered_groups.return_value = [
            SGLangKVComponentGroup("full", tensors, rope_num_layers=1)
        ]
        adapter.is_mla_enabled.return_value = False
        module_name = "lmcache.integration.sglang.unified_kv_adapter"
        adapter_module = ModuleType(module_name)
        vars(adapter_module)["SGLangUnifiedKVAdapter"] = Mock(return_value=adapter)
        config = SimpleNamespace(mp_host="127.0.0.1", mp_port=5555)
        config.get_extra_config_value = lambda _key, default: default
        self.client = Mock()
        self.client.get_chunk_size.return_value.result.return_value = 8
        self.client.cb_protocol_handshake.return_value.result.return_value = (2, True)
        self.future = self.client.cb_unified_lookup.return_value
        self.future.query.return_value = True
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
            patch("lmcache.v1.client.worker_transfer.create_transfer_context"),
        ):
            self.connector = UnifiedLMCacheMPConnector(
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
        self.addCleanup(self.connector.close)

    def test_sparse_only_hit_survives_pending_polls_and_finishes_once(self) -> None:
        result = CBUnifiedLookupResult(0, [CBMatchResult(0, 8, 8, 16, b"hash")])
        self.future.result.side_effect = [None, result]
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
        self.client.end_session.assert_called_once_with("sparse")
        with self.assertRaisesRegex(RuntimeError, "retired"):
            self.connector.poll_blend_lookup(operation)

    def test_releasing_prefix_keeps_sparse_and_segmented_matches(self) -> None:
        result = CBUnifiedLookupResult(
            8,
            [CBMatchResult(0, 8, 24, 32, b"shifted")],
            [CBMatchResult(16, 24, 16, 24, b"exact")],
        )
        self.future.result.return_value = result
        operation = self.connector.submit_blend_lookup(
            "mixed", list(range(32)), local_hit_tokens=8, cache_salt=""
        )
        self.assertIs(self.connector.poll_blend_lookup(operation), result)
        self.connector.free_lookup_locks("mixed", 0, 8)
        self.assertIs(self.connector.poll_blend_lookup(operation), result)
        self.assertEqual(operation.total_hit_tokens, 8)
        self.connector.finish_request("mixed")
        self.client.end_session.assert_called_once_with("mixed")

    def test_cancellation_does_not_poll_or_resubmit_pending_rpc(self) -> None:
        self.future.query.return_value = False
        operation = self.connector.submit_blend_lookup(
            "cancelled", list(range(16)), local_hit_tokens=0, cache_salt=""
        )
        self.assertIsNone(self.connector.poll_blend_lookup(operation))
        self.connector.finish_request("cancelled")
        self.future.query.return_value = True
        with self.assertRaisesRegex(RuntimeError, "retired"):
            self.connector.poll_blend_lookup(operation)
        self.future.result.assert_not_called()
        self.client.cb_unified_lookup.assert_called_once()
        self.client.end_session.assert_called_once_with("cancelled")

    def test_old_server_rejected_before_lookup(self) -> None:
        self.client.cb_protocol_handshake.return_value.result.return_value = (1, True)
        with self.assertRaisesRegex(RuntimeError, "protocol v2"):
            self.connector.submit_blend_lookup(
                "old", list(range(16)), local_hit_tokens=0, cache_salt=""
            )
        self.client.cb_unified_lookup.assert_not_called()
        self.connector.finish_request("old")
        self.client.end_session.assert_not_called()

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
