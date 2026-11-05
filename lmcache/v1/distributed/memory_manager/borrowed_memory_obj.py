# SPDX-License-Identifier: Apache-2.0
"""A tensor view of bytes another party owns, held under a lease."""

# Standard
from typing import Callable

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
            the caller might hold. It runs under the L1 manager's lock, on
            whichever thread frees the view, so it must never call back into
            an L1 manager and may only submit the peer unlock, never wait for
            it.
    """

    def __init__(
        self,
        raw_data: torch.Tensor,
        metadata: MemoryObjMetadata,
        *,
        lease_is_valid: Callable[[], bool],
        on_release: Callable[[], None],
    ) -> None:
        super().__init__(raw_data, metadata, parent_allocator=None)
        self._lease_is_valid = lease_is_valid
        self._on_release = on_release

    @property
    def lease(self) -> tuple[Callable[[], bool], Callable[[], None]]:
        """Return the view's ``(lease_is_valid, on_release)`` pair."""
        return self._lease_is_valid, self._on_release

    def replace_lease(
        self, lease_is_valid: Callable[[], bool], on_release: Callable[[], None]
    ) -> None:
        """Put the view under another lease, a newer reservation of the same
        bytes. The caller owns the previous lease.

        Args:
            lease_is_valid: Nonblocking check that the new lease still holds.
            on_release: Ends the new lease; same rules as the constructor's.
        """
        self._lease_is_valid = lease_is_valid
        self._on_release = on_release

    def same_bytes(self, other: "BorrowedTensorMemoryObj") -> bool:
        """Return whether ``other`` is a view of the same bytes as this one.

        Args:
            other: A view of the same key from a later lookup.

        Returns:
            True when both views alias one buffer of one size. A lender whose
            views land in local buffers overrides this to compare the peer
            address instead.
        """
        return (
            self.raw_data.data_ptr() == other.raw_data.data_ptr()
            and self.get_size() == other.get_size()
        )

    def is_valid(self) -> bool:
        """Return whether the view is unreleased and its lease still holds."""
        return super().is_valid() and self._lease_is_valid()

    def release(self) -> None:
        """Invalidate the view and end the lease; repeated calls are no-ops."""
        self.invalidate()
        self._on_release()
