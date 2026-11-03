# SPDX-License-Identifier: Apache-2.0
"""Shadow L1 manager: indexes borrowed views whose bytes another party owns."""

# Standard
from collections.abc import Mapping
from dataclasses import replace

# First Party
from lmcache import torch_dev
from lmcache.lmcache_native import TTLLock
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ObjectMeta, L1OperationResult
from lmcache.v1.distributed.l1_manager.base import (
    BaseL1Manager,
    L1ObjectState,
    l1_mgr_synchronized,
    validate_read_locks,
)
from lmcache.v1.distributed.memory_manager import ShadowL1MemoryManager
from lmcache.v1.memory_management import BorrowedTensorMemoryObj, MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType

# The L1 medium a borrowed segment is reported as: a peer's DRAM view counts
# as DRAM, a peer's CXL view as Device-DAX. Segments no L1 backs cannot be
# borrowed.
_BACKEND_BY_SEGMENT: dict[MemorySegment, L1BackendType] = {
    MemorySegment.DRAM: L1BackendType.DRAM,
    MemorySegment.CXL: L1BackendType.DEVDAX,
    MemorySegment.GDS: L1BackendType.GDS,
}


class ShadowL1Manager(BaseL1Manager):
    """L1 manager whose objects are borrowed views: metadata only, no payload.

    An L2 adapter that lends views (e.g. of a peer's memory) produces
    :class:`BorrowedTensorMemoryObj` objects; this manager indexes them by key
    so the ordinary read path (read locks, ``finish_read``, events) serves
    them. Freeing an object ends its lease instead of returning memory. The
    manager has no capacity, so write placement never targets it, and it runs
    no eviction controller: the peer that owns the bytes evicts them, and
    views leave here when their last reader finishes or
    :meth:`reap_expired` collects them.

    A view whose lease ended is never handed to a new reader. A reader that
    already holds it keeps it until its last ``finish_read`` (views are
    temporary), so a copy in flight is never torn down.
    """

    def __init__(
        self,
        config: L1ManagerConfig,
        segment: MemorySegment,
        backend_type: L1BackendType,
    ) -> None:
        """Create a shadow L1 for views of one memory segment.

        Args:
            config: Lifetime and reporting settings for the shadow L1.
            segment: The segment the borrowed bytes live in, reported as
                REMOTE.
            backend_type: The L1 medium the views are reported as.
        """
        self._backend_type = backend_type
        self._memory_tags = MemoryTags(
            segment=segment, residency=MemoryResidency.REMOTE
        )
        super().__init__(config, ShadowL1MemoryManager())

    @classmethod
    def for_lender(
        cls, adapter_id: int, segment: MemorySegment, defaults: L1ManagerConfig
    ) -> "ShadowL1Manager":
        """Create the shadow L1 a lending L2 adapter brings with it.

        The adapter resolves and maps the peers' memory, like the P2P
        workflow; nothing about the borrowed bytes is known in advance, so
        the shadow needs no size or device. It takes the lock TTLs from
        ``defaults``, is tagged ``shadow.<segment>.<adapter_id>``, and has no
        memory of its own.

        Args:
            adapter_id: Stable id of the lending adapter; makes the tag unique.
            segment: The segment the adapter borrows (its wanted L1 tags).
            defaults: The L1 settings the shadow inherits TTLs from.

        Returns:
            A new, open shadow L1 manager.

        Raises:
            ValueError: If no L1 medium backs ``segment``.
        """
        backend_type = _BACKEND_BY_SEGMENT.get(segment)
        if backend_type is None:
            raise ValueError(f"No L1 medium backs borrowed {segment.value} views")
        config = replace(
            defaults,
            tag=f"shadow.{segment.value}.{adapter_id}",
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=0, use_lazy=False, shm_name=""
            ),
            gds_l1_config=None,
            eviction=None,
        )
        return cls(config, segment, backend_type)

    @property
    def backend_type(self) -> L1BackendType:
        """Return the L1 medium the borrowed views are reported as."""
        return self._backend_type

    def get_memory_tags(self) -> MemoryTags:
        """Return the borrowed segment with REMOTE residency."""
        return self._memory_tags

    @l1_mgr_synchronized
    def admit(
        self,
        objects: Mapping[ObjectKey, BorrowedTensorMemoryObj],
        read_locks: int = 1,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Make borrowed views resident and read-lock them, without allocating.

        The view's ownership passes to this manager: every view not kept
        resident is released before returning. A resident view whose lease
        ended and that no reader holds is replaced by the given view.

        Args:
            objects: Borrowed view per key.
            read_locks: Read locks to take per admitted key, one per reader.

        Returns:
            Per key: ``(SUCCESS, obj)`` with the resident object now
            read-locked -- the given view, or an already-resident one, in
            which case the given view is released; ``(KEY_NOT_READABLE,
            None)`` when the resident view's lease ended while readers still
            hold it; ``(KEY_IN_WRONG_STATE, None)`` when the given view's
            lease ended before admission.

        Raises:
            ValueError: If ``read_locks`` is out of range.
        """
        total = validate_read_locks(read_locks)
        ret: dict[ObjectKey, L1OperationResult] = {}
        admitted_keys: list[ObjectKey] = []
        admitted_meta: list[L1ObjectMeta] = []
        resident_keys: list[ObjectKey] = []
        discarded: list[MemoryObj] = []
        replaced_keys: list[ObjectKey] = []
        replaced: list[MemoryObj] = []
        for key, obj in objects.items():
            entry = self._objects.get(key)
            if (
                entry is not None
                and not entry.memory_obj.is_valid()
                and not entry.read_lock.is_locked()
            ):
                # Stale and unheld: the reaper would collect it; do so now
                # so the fresh view is not refused behind it.
                replaced_keys.append(key)
                replaced.append(self._objects.pop(key).memory_obj)
                entry = None
            if entry is not None:
                discarded.append(obj)
                if not entry.memory_obj.is_valid():
                    ret[key] = (L1Error.KEY_NOT_READABLE, None)
                    continue
                resident_keys.append(key)
            elif not obj.is_valid():
                discarded.append(obj)
                ret[key] = (L1Error.KEY_IN_WRONG_STATE, None)
                continue
            else:
                entry = L1ObjectState(
                    memory_obj=obj,
                    write_lock=TTLLock(self._write_ttl_seconds),
                    read_lock=TTLLock(self._read_ttl_seconds),
                    is_temporary=True,
                )
                obj.set_l1_manager(self.l1_manager_id)
                self._objects[key] = entry
                admitted_keys.append(key)
                admitted_meta.append(self._object_meta(obj))
            for _ in range(total):
                entry.read_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)

        if replaced_keys:
            self._free_and_report_deleted(replaced_keys, replaced)
        self._memory_manager.free(discarded)
        if resident_keys:
            self._report_read_reserved(resident_keys)
        for listener in self._registered_listeners:
            listener.on_l1_keys_finish_write_and_reserve_read(admitted_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
                metadata={
                    "l1_tag": self._config.tag,
                    "keys": admitted_keys,
                    "meta": admitted_meta,
                },
            )
        )
        return ret

    @l1_mgr_synchronized
    def reap_expired(self) -> None:
        """Release every view no reader holds.

        A view is unheld once its last ``finish_read`` ran (which frees it
        at once) or its read lock's TTL lapsed, in which case a late
        ``finish_read`` is refused and would never free it. The lease state
        does not matter: an unheld view serves nobody, and a reader past its
        TTL is treated as gone, as in every L1. A held view whose lease ended
        stays, unreadable to new readers, until its last ``finish_read``.
        """
        expired = [
            key
            for key, entry in self._objects.items()
            if not entry.read_lock.is_locked()
        ]
        if expired:
            objs = [self._objects.pop(key).memory_obj for key in expired]
            self._free_and_report_deleted(expired, objs)

    def release_all(self) -> None:
        """Release every view, read-locked or not, before lenders shut down.

        Call after stopping new controller and GPU submissions; outstanding
        work on every device is synchronized first so no copy reads a
        released view.
        """
        if torch_dev.is_available():
            for device in range(torch_dev.device_count()):
                torch_dev.synchronize(device)
        self.clear(force=True)

    def _object_meta(self, memory_obj: MemoryObj) -> L1ObjectMeta:
        """Report borrowed views as holding no local bytes."""
        return L1ObjectMeta(size_bytes=0, backend=self.backend_type)
