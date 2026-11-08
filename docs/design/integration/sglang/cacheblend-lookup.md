# Request-owned CacheBlend lookup and retrieve

The explicit `submit_blend_lookup` / `poll_blend_lookup` API prepares matches
for selective prefill. It is not enabled by the existing unified radix-cache
scheduler. Lookup accepts full-MHA tensor-parallel registrations; pipeline
parallelism remains unsupported. It requires a compatible base blend protocol
plus the `blend_rotary_components` feature from `GET_EXPERIMENTAL`. The feature confirms separate rotary and value
components; a newer protocol version alone does not. Both checks run before
the first blend lookup. Sparse transfer and rotary registration still require a
single rank until cross-rank transfer completion is supported.

## Ownership and prefix semantics

`LMCacheBlendLookupOperation` owns the complete `CBUnifiedLookupResult`.
Its inherited `total_hit_tokens` contains only `prefix_coverage_tokens`.
Non-prefix matches require checking/recomputation; segmented-prefix matches
are exact cached chunks beyond a gap. Neither kind extends the contiguous
prefix reported to the radix cache.

For example, prefix coverage `[0, 256)` and a shifted match `[512, 768)`
report 256 prefix tokens. The missing `[256, 512)` still needs computation.
Releasing locks for `[0, 256)` preserves the request's sparse match metadata.

## RPC lifecycle

```
submit_blend_lookup -> CB_UNIFIED_LOOKUP
poll_blend_lookup  -> pending future / pending server prefetch / complete result
finish_request     -> retire client operation -> last lookup reply
                   -> prefix releases acknowledged -> END_SESSION acknowledged
```

A pending server result is polled with a fresh RPC on a later scheduler pass.
Completed results remain on the operation, so repeated client polls do not
start another server lookup. Retired operations cannot be polled. Empty aligned
prompts submit no lookup and need no session-ending RPC.

Each blend operation has a unique server `session_id`; `request_id` continues
to identify the engine request. Lookup, prefix and sparse retrieves, explicit
lock releases, stores, and END_SESSION use that same server identity. An engine
ID reused after `finish_request` gets a new session, so delayed cleanup cannot
remove the replacement's state. No history of retired IDs or scheduler wait is
needed. Submit the blend lookup before any stores, including for short prompts
whose initial lookup submits no RPC; a later store uses that operation's session.

Server teardown disposes pending prefetch results and releases the prefix
reservations of unfinished lookups. Client retirement waits asynchronously for
the last lookup RPC before sending END_SESSION, so a late lookup cannot create
its session after teardown. If that reply contains a completed prefix, retirement
releases the prefix range that no retrieve or explicit release consumed. Earlier
explicit prefix releases must also acknowledge before END_SESSION: their server
handler needs the session's recorded lock owners.

The scheduler retires local state immediately. Future callbacks advance the RPC
chain without blocking or editing scheduler dictionaries. One tracked future
covers the entire chain, so `close` waits for END_SESSION's acknowledgement up
to the configured RPC timeout before closing the transport. RPC failures are
logged and teardown is still attempted; a disconnected server cannot guarantee
remote cleanup. Callers must finish submitted retrieves before retiring the
request, because H2D completion owns their lock release.

The client tests establish RPC and metadata ownership, not that selective
generation is complete. Multi-rank model execution, transfer completion and PP
match distribution remain separate integration work.

## Tensor-parallel lookup agreement

All TP ranks submit and poll requests in the same order. Only TP rank zero
negotiates the server capabilities and issues lookup RPCs. It broadcasts one
session ID and the complete lookup result through the connector's CPU TP group;
non-prefix and segmented-prefix spans remain distinct. All ranks observe the
same pending/completed state and contiguous store cursor. RPC errors are agreed
before broadcasting a result: the failing rank raises its original exception,
and peers raise `RuntimeError`. Call `finish_request` on every rank after a failure,
including submission failures for which no operation was returned.

Only the leader releases prefix locks and sends `END_SESSION`. Its existing
callback chain waits for an outstanding lookup reply before retiring the shared
session. Other ranks retire their local operation immediately. A subsequent
request reusing the engine ID receives a new shared server ID, so a late reply
cannot end the new session.

This enables TP lookup metadata, not TP sparse inference. Rotary registration
and sparse retrieve explicitly reject multiple ranks before submitting device
work. Transfer completion must be agreed across ranks before removing that gate;
a rank-local success must not choose sparse execution while a peer recomputes.

## Explicit rotary registration and sparse retrieve

`register_blend_rope(table, is_neox_style=...)` registers a model-owned rotary
cache shared by the full-MHA layers. The table contains concatenated cosine/sine
halves; its width is the rotary dimension, while each group's head dimension
comes from the registered KV shape. `None` represents NoPE. Distinct per-layer
rotary tables require a different grouping contract and are not represented by
this shared-table entry point.

After a completed lookup, `submit_blend_retrieve` accepts full request slot
tables, padded to physical pages, plus an optional subset of prefetched matches.
It includes segmented-prefix matches by default. It requires the destination to
cover every selected match and rejects padding/out-of-range destination pages.
This prevents interpreting the server's beyond-allocation deferral as completed
KV population. Block tables remain in **engine group** order; ordinary transfer
helpers expand tables to kernel-group order and must not be used here.

The returned future orders KV writes after a producer event and supports
`wait_on_stream` before selective model execution. Retain the future and the
allocation until device completion, then finish the request. Sparse completion
does not consume contiguous-prefix locks; `complete_load` applies only to the
separate prefix retrieve. Selecting no matches returns a completed future and
leaves cleanup to request retirement.

These methods are opt-in connector plumbing. Normal SGLang scheduling still
needs to select recomputation tokens, restore residuals, and keep retrieved
slots alive throughout selective execution. A successful transfer does not
establish generation quality or a latency improvement.

## Chunked retrieval

Pass the allocated slot table from token zero through the current window and
select only matches wholly covered by that window. When lookup matches remain
beyond the allocated table, the connector requires protocol v2 before creating
a device event or submitting a transfer. The explicit `blend_rotary_components`
capability is still required: chunked lock retention and rotary support are
independent contracts.

Complete each retrieve future before submitting the next window, use increasing
windows, and send each match once. An unmatched prompt suffix does not require
v2 when every lookup match already fits: no later reservation needs retaining.
This connector API does not itself enable chunked selective prefill in SGLang's
scheduler.
