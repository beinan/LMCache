# SPDX-License-Identifier: Apache-2.0
"""A mock L2 adapter that lends views; importing registers ``lending_mock``.

It satisfies the ``LendingL2Adapter`` protocol without being a
``PeerL2AdapterBase``, so controller tests need no transport.
"""

# Standard
from typing import Optional

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.memory_management import BorrowedTensorMemoryObj


class LendingMockL2AdapterConfig(MockL2AdapterConfig):
    """Config for a mock adapter that lends views of a peer's DRAM.

    It wants a ``(DRAM, REMOTE)`` L1, so the storage manager gives it a
    shadow L1 instead of binding it to a local L1.
    """

    def get_wanted_l1_memory_tags(self) -> tuple[MemoryTags, ...]:
        return (MemoryTags(MemorySegment.DRAM, MemoryResidency.REMOTE),)


class LendingMockL2Adapter(MockL2Adapter):
    """Mock adapter that lends views of its stored objects .

    A view's lease is the key's lookup lock: ending it unlocks the key.
    ``lookup_count`` counts lookups, so a test can assert that a request
    never consulted the lender; ``get_active_borrow_count`` counts views
    whose lease has not ended, so adapter removal drains them first.
    """

    def __init__(self, config: MockL2AdapterConfig) -> None:
        super().__init__(config)
        self._lent: dict[L2TaskId, set[ObjectKey]] = {}
        self._live_views = 0
        self.lookup_count = 0

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
    ) -> L2TaskId:
        self.lookup_count += 1
        return super().submit_lookup_and_lock_task(keys, group_layout_descs)

    def take_borrowed_objects(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        layouts: dict[int, MemoryLayoutDesc],
    ) -> dict[ObjectKey, BorrowedTensorMemoryObj]:
        self._lent[task_id] = set(keys)
        views = {}
        for key in keys:
            stored = self._memory_objects[key]
            ended: list[bool] = []
            self._live_views += 1

            def lease_is_valid(ended: list[bool] = ended) -> bool:
                return not ended

            def end_lease(key: ObjectKey = key, ended: list[bool] = ended) -> None:
                if not ended:
                    ended.append(True)
                    self._live_views -= 1
                    self.submit_unlock([key])

            views[key] = BorrowedTensorMemoryObj(
                stored.raw_data,
                stored.meta,
                lease_is_valid=lease_is_valid,
                on_release=end_lease,
            )
        return views

    def release_lookup(self, task_id: L2TaskId, keys: list[ObjectKey]) -> None:
        lent = self._lent.pop(task_id, set())
        self.submit_unlock([key for key in keys if key not in lent])

    def get_active_borrow_count(self) -> int:
        return self._live_views


def _create_lending_mock_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: Optional[L1MemoryDesc] = None,
) -> L2AdapterInterface:
    """Create a LendingMockL2Adapter from config."""
    return LendingMockL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_type("lending_mock", LendingMockL2AdapterConfig)
register_l2_adapter_factory("lending_mock", _create_lending_mock_adapter)
