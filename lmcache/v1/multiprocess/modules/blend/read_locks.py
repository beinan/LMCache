# SPDX-License-Identifier: Apache-2.0
"""The sparse lookup's read-lock reservation and its per-request ledger."""

# Standard
from collections import Counter
from dataclasses import dataclass, field

# First Party
from lmcache.v1.distributed.api import (
    ObjectKey,
    ShardReplicationDesc,
    ipc_key_to_object_keys,
)
from lmcache.v1.multiprocess.custom_types import CBMatchResult, IPCCacheServerKey


@dataclass
class ReadLockReservation:
    """The read locks one request's sparse lookup took, and which it still holds.

    The lookup read-locks every object key of every match it finds,
    once per reader of that object group and token shard. A request may then be
    retrieved over several calls, one per engine prefill chunk, so the
    reservation lives on the session. ``held`` counts, per key, the locks the
    request still holds and nobody has claimed. Every lock leaves it exactly
    once: a retrieve claims one per key it reads (and then owns its release),
    a sweep releases a match no later call can send, and session end releases
    the rest. A key whose count is 0 is never read or released again.

    Not thread-safe: the owner (``BlendModule``) guards every call with
    ``_cb_retain_lock``.
    """

    #: Registration used by the lookup to map ranks and count readers.
    shard_replication_desc: ShardReplicationDesc
    #: Every rank's object keys per match hash, group-major and rank-minor.
    per_hash: dict[bytes, list[ObjectKey]]
    #: Where each match ends in the prompt (``CBMatchResult.cur_ed``).
    ends: dict[bytes, int]
    #: The L1 instance holding each key (multi-L1), or None.
    l1_owners: dict[ObjectKey, int] | None = None
    #: Locks still held per key, unclaimed.
    held: dict[ObjectKey, int] = field(init=False)

    def __post_init__(self) -> None:
        self.held = {
            k: self.shard_replication_desc.num_kv_readers(k.object_group_id)
            for keys in self.per_hash.values()
            for k in keys
        }

    def rank_keys(
        self,
        cb_match_result: list[CBMatchResult],
        key: IPCCacheServerKey,
        blend_gids: tuple[int, ...],
    ) -> list[ObjectKey] | None:
        """This rank's reserved keys for every match, chunk-major.

        A retrieve reads only keys its request's lookup locked, so a match the
        lookup did not reserve is never given keys to read.

        :param cb_match_result: The matches a retrieve consumes.
        :param key: The retrieve's cache key (``worker_id`` / ``world_size``).
        :param blend_gids: Object group ids read by the sparse leg.
        :return: One key per read group per match, in match order, or None when any
            match is not in the reservation.
        :raises ValueError: If a reserved match lacks a key required by this
            rank, i.e. the lookup and the retrieve disagree on the layout.
        """
        out: list[ObjectKey] = []
        for r in cb_match_result:
            reserved = self.per_hash.get(r.hash)
            if reserved is None:
                return None
            if set(blend_gids) != {k.object_group_id for k in reserved}:
                raise ValueError(
                    "CB retrieve: the lookup did not reserve the read groups."
                )
            per_group = ipc_key_to_object_keys(
                key, [r.hash], list(blend_gids), self.shard_replication_desc
            )
            if any(len(keys) != 1 or keys[0] not in reserved for keys in per_group):
                raise ValueError(
                    "CB retrieve: the lookup did not reserve this rank's "
                    "key for every read group."
                )
            out.extend(keys[0] for keys in per_group)
        return out

    def claim(self, keys: list[ObjectKey]) -> bool:
        """Claim one held lock per occurrence of each key, all or nothing.

        :param keys: The keys one retrieve reads.
        :return: True if every key had a lock left for each of its
            occurrences; the caller then owns those locks and must release
            them. False (and nothing claimed) otherwise.
        """
        need = Counter(keys)
        if any(self.held.get(k, 0) < n for k, n in need.items()):
            return False
        for k, n in need.items():
            self.held[k] -= n
        return True

    def sweep(
        self, sent: set[bytes], upto: int, final: bool
    ) -> dict[int, list[ObjectKey]]:
        """Release the matches no later retrieve can send.

        A client sends each match at most once, in the call whose window
        holds it whole, and calls for a request in increasing window order.
        So a match this call was not sent is unreachable once it ends at or
        before the last match this call was sent (``upto``), and every unsent
        match is unreachable on the final call (``final``).

        :param sent: Hashes this retrieve was sent.
        :param upto: The largest ``cur_ed`` among them.
        :param final: Whether the engine has allocated the whole prompt.
        :return: The locks to release, grouped by count per key.
        """
        out: dict[int, list[ObjectKey]] = {}
        for h in [h for h in self.per_hash if h not in sent]:
            if not final and self.ends[h] > upto:
                continue
            for k in self.per_hash.pop(h):
                n = self.held.pop(k, 0)
                if n:
                    out.setdefault(n, []).append(k)
        return out

    def release_all(self) -> dict[int, list[ObjectKey]]:
        """Give up every lock still held (session end, superseded lookup).

        :return: The locks to release, grouped by count per key.
        """
        out: dict[int, list[ObjectKey]] = {}
        for k, n in self.held.items():
            if n:
                out.setdefault(n, []).append(k)
        self.held.clear()
        self.per_hash.clear()
        return out
