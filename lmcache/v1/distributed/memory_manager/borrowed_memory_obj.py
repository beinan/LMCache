# SPDX-License-Identifier: Apache-2.0
"""A tensor view of bytes another party owns, held under a lease."""

# Standard
from typing import Callable, Optional

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj


class BorrowedTensorMemoryObj(TensorMemoryObj):
    """A tensor view of bytes another party owns, held under a lease.

    It allocates nothing locally. Whoever lends the bytes (e.g. an L2 adapter
    holding a peer's read reservation) supplies the lease: ``is_valid()``
    also reports whether the lease still holds, and :meth:`release` ends it.
    A lender must release a view that is garbage-collected without
    :meth:`release`.

    Args:
        raw_data: The borrowed bytes; a view, never a copy.
        metadata: Layout of the object.
        lease_is_valid: Nonblocking check that the lease still holds.
        on_release: Ends the lease; idempotent, nonblocking, and takes no lock
            the caller might hold. It runs under the L1 manager's lock, so it
            must never call back into an L1 manager.
        on_touch: Optional; called when a reader hits the view, so the lender
            can renew a lease that is near its end. Same rules as
            ``on_release``; it may also run after :meth:`release` and must
            then do nothing.
    """

    def __init__(
        self,
        raw_data: torch.Tensor,
        metadata: MemoryObjMetadata,
        *,
        lease_is_valid: Callable[[], bool],
        on_release: Callable[[], None],
        on_touch: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(raw_data, metadata, parent_allocator=None)
        self._lease_is_valid = lease_is_valid
        self._on_release = on_release
        self._on_touch = on_touch

    def is_valid(self) -> bool:
        """Return whether the view is unreleased and its lease still holds."""
        return super().is_valid() and self._lease_is_valid()

    def release(self) -> None:
        """End the lease; repeated calls are no-ops."""
        self._on_release()

    def touch(self) -> None:
        """Tell the lender a reader hit the view, if it asked to know."""
        if self._on_touch is not None:
            self._on_touch()
