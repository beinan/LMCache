# SPDX-License-Identifier: Apache-2.0
"""The contract of an L2 adapter that lends a peer's bytes instead of copying."""

# Future
from __future__ import annotations

# Standard
from abc import abstractmethod
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.memory_management import BorrowedTensorMemoryObj


class LendingL2Adapter(L2AdapterInterface):
    """An L2 adapter that hands L1 views of a peer's bytes instead of copies.

    A shadow L1 is supported only with such an adapter: the storage manager
    creates one for it at registration, and the prefetch controller admits
    the views it lends there. The production lenders are peer adapters
    (``PeerL2AdapterBase`` subclasses that also derive from this class); they
    want a ``REMOTE`` L1 and resolve the peer's memory themselves.
    """

    @abstractmethod
    def take_borrowed_objects(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        layouts: dict[int, MemoryLayoutDesc],
    ) -> dict[ObjectKey, BorrowedTensorMemoryObj]:
        """Lend views of found keys instead of copying them.

        Args:
            task_id: Completed lookup's local task identity.
            keys: Selected keys whose reservations the caller will own.
            layouts: Expected layout by object-group ID.

        Returns:
            A borrowed view per key it can lend; each view's lease covers the
            key's reservation. Missing keys stay releasable through
            :meth:`release_lookup`.

        Note:
            Must not raise once it has created a view; a key it cannot lend
            is omitted, since a view that never reaches the caller is only
            released by its finalizer.
        """

    @abstractmethod
    def release_lookup(self, task_id: L2TaskId, keys: list[ObjectKey]) -> None:
        """Return the reservations of a lookup that were not moved into views.

        Args:
            task_id: Local lookup identity.
            keys: Reservations to return.
        """

    @abstractmethod
    def get_active_borrow_count(self) -> int:
        """Return live borrowed views, which must drain before removal."""
