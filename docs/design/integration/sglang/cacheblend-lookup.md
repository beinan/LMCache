# Request-owned CacheBlend lookup

The explicit `submit_blend_lookup` / `poll_blend_lookup` API prepares matches
for selective prefill. It is not enabled by the existing unified radix-cache
scheduler. It currently accepts single-rank full MHA registrations and requires
server protocol v2, which supports separate rotary and value components.

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
generation is complete. Retrieval events, model execution, and TP/PP match
distribution remain separate integration work.
