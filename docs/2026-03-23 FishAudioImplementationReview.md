Fish Audio Integration Review
Fish Audio S2 Pro was added across two commits (138828a, ffbffd7), touching 5 areas: a new FishAudioService wrapper, a dedicated Dockerfile.sglang, compose additions, new settings, and a /v1/tts/s2-pro/sync endpoint.

Build Process Issues
1. Unpinned upstream dependency — highest risk
Dockerfile.sglang:11: ARG SGLANG_OMNI_REF=main clones sglang-omni from the main branch with no commit SHA. Any upstream push silently changes your build. Should pin to a specific commit or tag.

2. Fragile string-patch strategy
Lines 85–229 apply 5 source-code patches to installed site-packages files using exact string matching. Each patch calls raise SystemExit(...) if the expected string isn't found — so a single upstream whitespace change breaks the entire build with no recovery. These patches also span 7 separate RUN python - blocks, each adding a new layer unnecessarily. They could be merged into 1–2 RUN steps.

3. No healthcheck on fish-sglang
compose.yaml:155-183: The fish-sglang service has no healthcheck, so other services can't declare condition: service_healthy dependency on it. The API container will forward S2Pro requests to SGLang immediately on startup even if the model hasn't finished loading.

4. Missing depends_on for fish-sglang
echofleet-qwen3-tts depends on redis but not fish-sglang. If SGLang is slow to start (likely, since it downloads model weights on first run), the API will return 500s until SGLang is ready.

5. Inconsistent volume mount mode
compose.yaml:82: fish-tts:/models_fish-s2-pro in the API container is writable, but compose.yaml:146 mounts it :ro in the worker. Should be :ro in both — the API container doesn't write to the model volume.

6. Missing env vars in worker
compose.yaml:136-147: The worker has S2_PRO_MODEL_DIR but is missing SGLANG_BASE_URL and FISH_REF_AUDIO_DIR. If the worker ever dispatches Fish Audio jobs, it will use hardcoded defaults from settings.py instead of compose-configured values.

Performance Issues
7. Stage 2 decoder pinned to CPU
compose.yaml:180: --stages.2.executor.args.device cpu forces the vocoder/decoder stage to CPU. This was likely a flash-attn compatibility workaround, but now that the SDPA KV-cache fallback is baked in (Dockerfile.sglang lines 115–218), it may be possible to move stage 2 back to CUDA, which would significantly reduce latency.

8. disable_cuda_graph=True
Dockerfile.sglang:224-229: CUDA graphs are disabled. They can reduce per-step overhead for autoregressive decoding by 20–40%. This was likely disabled due to initial compatibility issues and is worth re-enabling (or testing) once the build is stable.

9. FISH_FORCE_SDPA_KVCACHE=true by default
Dockerfile.sglang:126-133: The env var defaults to true, always bypassing flash-attn for KV-cache decode. The slower SDPA path (lines 179–211) does a Python loop over the batch and a manual repeat_interleave GQA expansion. On supported GPUs, flash-attn-kvcache is substantially faster. The env var should default to false and only be set to true as a fallback.

10. Blocking call in async endpoint
api/main.py:1842: svc.generate(request.text, request.language) is a synchronous requests.post() with a 5-minute timeout, called directly inside an async def endpoint. This blocks the FastAPI event loop for the entire generation duration. It should be wrapped in loop.run_in_executor(None, ...).

11. No HTTP connection reuse
inference/fish_audio_service.py:73: requests.post(endpoint, ...) creates a new connection per request. A requests.Session stored on the service instance would reuse TCP connections and reduce per-request overhead.

12. Unused imports
inference/fish_audio_service.py:8-11: gc, base64, and json are imported but never used — likely leftovers from an earlier draft.

Summary Priority Table
Priority	Issue	Fix
High	Unpinned SGLANG_OMNI_REF=main	Pin to a commit SHA/tag
High	Blocking svc.generate() in async handler	Wrap in run_in_executor
High	No fish-sglang healthcheck or depends_on	Add healthcheck + dependency
Medium	Stage 2 CPU pinning	Test moving to GPU now that SDPA fallback exists
Medium	FISH_FORCE_SDPA_KVCACHE defaults to true	Default to false; only set on incompatible GPUs
Medium	7 separate patch RUN layers	Consolidate to 1–2 RUN blocks
Medium	Fragile string-patch strategy	Long-term: fork/vendor sglang-omni patch
Low	No Session for HTTP reuse	Use requests.Session on the service
Low	Writable fish-tts mount in API container	Add :ro
Low	Missing worker env vars	Add SGLANG_BASE_URL, FISH_REF_AUDIO_DIR to worker
Low	Unused imports (gc, base64, json)	Remove them