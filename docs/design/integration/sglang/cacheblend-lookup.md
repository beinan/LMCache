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
finish_request     -> retire client operation and send END_SESSION once
```

A pending server result is polled with a fresh RPC on a later scheduler pass.
Completed results remain on the operation, so repeated client polls do not
start another server lookup. Retired operations cannot be polled. Empty aligned
prompts submit no lookup and need no session-ending RPC.

Before enabling the scheduler caller, server cancellation must also drain
pending prefetch handles and release their locks: END_SESSION currently cleans
completed sparse reservations but leaves pending CacheBlend jobs behind.
The client tests establish RPC and metadata ownership, not that server cleanup
or selective generation is complete. Retrieval events, model execution, and
TP/PP match distribution remain separate integration work.
