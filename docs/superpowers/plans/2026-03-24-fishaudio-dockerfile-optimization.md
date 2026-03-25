# Fish Audio Dockerfile Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the Fish Audio patch RUN layers in Dockerfile.sglang without changing runtime behavior first, then run a separate gated experiment to remove the explicit CUDA-graph disabling override only if validation passes.

**Architecture:** Phase 1 merges the four Python patch RUN blocks into a single layer while preserving the current `disable_cuda_graph=True` behavior. Phase 2 is a separate experiment that removes the override and restores the upstream default only after repeated runtime validation on the pinned `sglang-omni` commit.

**Tech Stack:** Docker, Dockerfile.sglang

---

## File Structure

- Modify: `Dockerfile.sglang` - consolidate the four Fish Audio patch `RUN` blocks into one block in Chunk 1, then optionally replace only the `stages.py` mutation with a verification-only check in Chunk 3.
- Validate against: `compose.yaml` - confirm the Fish Audio service still runs with `--stages.2.executor.args.device cuda`; no source edits are required in this file.

## Chunk 1: Consolidate Patch RUN Layers Without Changing Behavior

### Task 1: Consolidate patch RUN layers and preserve the current CUDA-graph override

**Files:**
- Modify: `Dockerfile.sglang:80-232`

- [ ] **Step 1: Review current patch structure**

Current structure has 4 separate RUN blocks:
- Lines 80-90: Patch 1 - attention backend (triton instead of fa3)
- Lines 92-113: Patch 2 - runtime import (add S2ProSGLangOutputProcessor)
- Lines 115-220: Patch 3 - SDPA KV-cache fallback (add FISH_FORCE_SDPA_KVCACHE env var handling)
- Lines 222-232: Patch 4 - explicitly disable CUDA graphs (True instead of upstream False)

- [ ] **Step 2: Treat patch mismatches as pinned-ref drift, not generic upstream drift**

`Dockerfile.sglang` pins `SGLANG_OMNI_REF` to a specific commit SHA. If one of the exact-string patches fails, first assume the plan is stale relative to that pinned commit or the pinned ref has been intentionally updated in this repo. Do not describe the first failure mode as an arbitrary upstream change.

- [ ] **Step 3: Replace the 4 RUN blocks with one consolidated patch block that keeps `disable_cuda_graph=True`**

Replace lines 80-232 with:
```dockerfile
# Apply all sglang-omni patches in a single layer for better build efficiency
# Preserve current runtime behavior in this commit; do not change CUDA-graph behavior here.
RUN python - <<'PY'
from pathlib import Path

# Patch 1: Attention backend - use triton instead of fa3 for compatibility
factory = Path('/opt/venv/lib/python3.12/site-packages/sglang_omni/models/fishaudio_s2_pro/factory.py')
text = factory.read_text(encoding='utf-8')
old = '    if server_args.attention_backend is None:\n        server_args.attention_backend = "fa3"\n'
new = '    if server_args.attention_backend is None:\n        server_args.attention_backend = "triton"\n'
if old not in text:
    raise SystemExit('Expected attention backend snippet not found in factory.py')
factory.write_text(text.replace(old, new, 1), encoding='utf-8')

# Patch 2: Runtime import - add S2ProSGLangOutputProcessor
text = factory.read_text(encoding='utf-8')
old = '''from .runtime.s2pro_sglang_ar import (
    S2ProSGLangIterationController,
    S2ProSGLangModelRunner,
    S2ProSGLangResourceManager,
)
'''
new = '''from .runtime.s2pro_sglang_ar import (
    S2ProSGLangIterationController,
    S2ProSGLangModelRunner,
    S2ProSGLangOutputProcessor,
    S2ProSGLangResourceManager,
)
'''
if old not in text:
    raise SystemExit('Expected runtime import block not found in factory.py')
factory.write_text(text.replace(old, new, 1), encoding='utf-8')

# Patch 3: SDPA KV-cache fallback - add FISH_FORCE_SDPA_KVCACHE env var handling
modeling = Path('/opt/venv/lib/python3.12/site-packages/sglang_omni/models/fishaudio_s2_pro/fish_speech/models/text2semantic/modeling.py')
text = modeling.read_text(encoding='utf-8')

old_flag = '''FISH_BATCH_INVARIANT = os.getenv("FISH_BATCH_INVARIANT", "false").lower() in (
    "true",
    "1",
    "yes",
)
'''
new_flag = old_flag + '''
# Default to false - flash-attn-kvcache is faster on supported GPUs
# Set to true only on GPUs where flash-attn kernel is unavailable
FISH_FORCE_SDPA_KVCACHE = os.getenv("FISH_FORCE_SDPA_KVCACHE", "false").lower() in (
    "true",
    "1",
    "yes",
)
'''
if old_flag not in text:
    raise SystemExit('Expected FISH_BATCH_INVARIANT snippet not found in modeling.py')
text = text.replace(old_flag, new_flag, 1)

old_block = '''        # Use flash_attn_with_kvcache - it handles KV cache update internally
        # q: (batch_size, seqlen, nheads, headdim)
        # k_cache/v_cache: (batch_size, seqlen_cache, nheads_k, headdim)
        # k/v: (batch_size, seqlen_new, nheads_k, headdim)
        k_cache, v_cache = self.kv_cache.get(bsz)
        y = flash_attn_kvcache_op(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=True,
            num_splits=1 if FISH_BATCH_INVARIANT else 0,
        )

        y = y.contiguous().view(bsz, seqlen, q_size)
'''

new_block = '''        # Prefer a portable SDPA fallback for KV-cache decode on GPUs where the
        # flash-attn kernel is unavailable.
        k_cache, v_cache = self.kv_cache.get(bsz)
        y = None

        if not FISH_FORCE_SDPA_KVCACHE:
            try:
                y = flash_attn_kvcache_op(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    k=k,
                    v=v,
                    cache_seqlens=cache_seqlens,
                    causal=True,
                    num_splits=1 if FISH_BATCH_INVARIANT else 0,
                )
            except RuntimeError as exc:
                if "no kernel image is available for execution on the device" not in str(exc):
                    raise
                log.warning("Falling back to SDPA KV-cache decode: %s", exc)

        if y is None:
            start_positions = cache_seqlens.to(torch.int64)
            for batch_idx in range(bsz):
                start = int(start_positions[batch_idx].item())
                end = start + seqlen
                k_cache[batch_idx, start:end] = k[batch_idx]
                v_cache[batch_idx, start:end] = v[batch_idx]

            total_lengths = start_positions + seqlen
            max_total = int(total_lengths.max().item())

            q = q.transpose(1, 2)
            full_k = k_cache[:bsz, :max_total].transpose(1, 2)
            full_v = v_cache[:bsz, :max_total].transpose(1, 2)
            full_k = full_k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
            full_v = full_v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

            key_positions = torch.arange(max_total, device=q.device)
            query_positions = start_positions[:, None] + torch.arange(seqlen, device=q.device)[None, :]
            valid_mask = key_positions.view(1, 1, max_total) <= query_positions.unsqueeze(-1)
            valid_mask = valid_mask & (key_positions.view(1, 1, max_total) < total_lengths.view(-1, 1, 1))
            attn_mask = valid_mask.unsqueeze(1)

            y = self._scaled_dot_product_attention(
                q,
                full_k,
                full_v,
                attn_mask=attn_mask,
                is_causal=False,
            )
            y = y.transpose(1, 2)

        y = y.contiguous().view(bsz, seqlen, q_size)
'''

if old_block not in text:
    raise SystemExit('Expected KV-cache block not found in modeling.py')
text = text.replace(old_block, new_block, 1)
modeling.write_text(text, encoding='utf-8')

# Preserve the current compatibility override during layer consolidation.
stages = Path('/opt/venv/lib/python3.12/site-packages/sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py')
text = stages.read_text(encoding='utf-8')
old = '        disable_cuda_graph=False,\n'
new = '        disable_cuda_graph=True,\n'
if old not in text:
    raise SystemExit('Expected disable_cuda_graph snippet not found in stages.py')
stages.write_text(text.replace(old, new, 1), encoding='utf-8')
PY
```

- [ ] **Step 4: Build the compose service image to verify the consolidated patch applies correctly**

Run: `docker compose -f compose.yaml build fish-sglang`
Expected: Build completes without patch errors, confirming the consolidated patch still matches the pinned `SGLANG_OMNI_REF`. Runtime behavior is validated separately in Chunk 2.

- [ ] **Step 5: If build fails, diagnose against the pinned ref**

If patch fails with "Expected ... snippet not found":
- Inspect the actual file contents installed from the pinned `SGLANG_OMNI_REF`. Because the normal build fails at the patch step, temporarily add a pre-patch inspection target to `Dockerfile.sglang`, build that target, inspect it, then remove the temporary target before continuing.

Temporarily make these Dockerfile edits:

- Change the first line to `FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS fish-sglang-base`
- Insert this block immediately before the first patch `RUN python - <<'PY'` block:

```dockerfile
FROM fish-sglang-base AS inspect-pinned-ref
ENTRYPOINT ["/opt/venv/bin/python"]

FROM fish-sglang-base
```

Then run:

```bash
docker build --target inspect-pinned-ref -t fish-sglang-inspect -f Dockerfile.sglang .
docker run --rm --entrypoint python fish-sglang-inspect - <<'PY'
from pathlib import Path

targets = {
    "factory.py": (
        Path('/opt/venv/lib/python3.12/site-packages/sglang_omni/models/fishaudio_s2_pro/factory.py'),
        [
            'server_args.attention_backend = "fa3"',
            'S2ProSGLangResourceManager',
        ],
    ),
    "modeling.py": (
        Path('/opt/venv/lib/python3.12/site-packages/sglang_omni/models/fishaudio_s2_pro/fish_speech/models/text2semantic/modeling.py'),
        [
            'FISH_BATCH_INVARIANT',
            'flash_attn_kvcache_op(',
        ],
    ),
    "stages.py": (
        Path('/opt/venv/lib/python3.12/site-packages/sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py'),
        [
            'disable_cuda_graph=False',
            'disable_cuda_graph=True',
        ],
    ),
}

for label, (path, needles) in targets.items():
    print(f"\n=== {label}: {path} ===")
    text = path.read_text(encoding='utf-8')
    for needle in needles:
        print(f"{needle}: {'FOUND' if needle in text else 'MISSING'}")
PY
```

After collecting the snippet status you need, remove the temporary `inspect-pinned-ref` target and restore the original first `FROM` line before continuing with the real patch edit.

- Update the patch strings only after confirming the pinned source differs from this plan
- If the failure is from `stages.py`, verify whether the pinned ref still defaults to `disable_cuda_graph=False` before changing the plan

- [ ] **Step 6: Commit the cleanup-only Dockerfile change**

```bash
# Stage only the Dockerfile change; do not include unrelated untracked files.
git add Dockerfile.sglang
git commit -m "build: consolidate Fish Audio patch layers"
```

---

## Chunk 2: Collect a Baseline Before Changing CUDA Graphs

### Task 2: Measure the current known behavior with CUDA graphs still disabled

**Files:**
- No source change required.

- [ ] **Step 1: Verify stage 2 is on GPU**

Check compose.yaml line 190:
```yaml
      --stages.2.executor.args.device cuda
```

- [ ] **Step 2: Build and start the current known-good image**

Use PowerShell on the Windows host:

```powershell
docker compose -f compose.yaml up -d --build fish-sglang

# Poll readiness instead of relying on a single early probe.
$ready = $false
for ($i = 1; $i -le 60; $i++) {
  try {
    curl.exe --fail --show-error "http://localhost:18001/v1/models" | Out-Null
    $ready = $true
    break
  }
  catch {
    Start-Sleep -Seconds 10
  }
}
if (-not $ready) {
  throw "fish-sglang did not become ready within 10 minutes"
}
```

Linux/bash equivalent:

```bash
docker compose -f compose.yaml up -d --build fish-sglang

# Poll readiness instead of relying on a single early probe.
ready=0
for i in {1..60}; do
  if curl --fail --show-error "http://localhost:18001/v1/models" > /dev/null; then
    ready=1
    break
  fi
  sleep 10
done
if [ "$ready" -ne 1 ]; then
  echo "fish-sglang did not become ready within 10 minutes" >&2
  exit 1
fi
```

Do not proceed to the experiment until `/v1/models` responds successfully.

- [ ] **Step 3: Run one warmup request and three measured requests**

Use PowerShell if you are running from the Windows host so the timings are easy to compare:

```powershell
# Warmup request
curl.exe --fail --show-error -X POST "http://localhost:18001/v1/audio/speech" `
  -H "Content-Type: application/json" `
  -d "{\"input\":\"Warmup request for Fish Audio S2 Pro.\",\"language\":\"english\"}" `
  --output warmup.wav

# Timed requests
1..3 | ForEach-Object {
  $n = $_
  $result = Measure-Command {
    curl.exe --fail --show-error -X POST "http://localhost:18001/v1/audio/speech" `
      -H "Content-Type: application/json" `
      -d "{\"input\":\"Timed request $n for Fish Audio S2 Pro.\",\"language\":\"english\"}" `
      --output "baseline-$n.wav"
  }
  [pscustomobject]@{ Request = $n; TotalSeconds = [math]::Round($result.TotalSeconds, 2) }
}

docker compose -f compose.yaml logs fish-sglang --tail 200
```

Linux/bash equivalent:

```bash
# Warmup request
curl --fail --show-error -X POST "http://localhost:18001/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"input":"Warmup request for Fish Audio S2 Pro.","language":"english"}' \
  --output warmup.wav

# Timed requests
for n in 1 2 3; do
  /usr/bin/time -f "request=${n} total_seconds=%e" \
    curl --fail --show-error -X POST "http://localhost:18001/v1/audio/speech" \
      -H "Content-Type: application/json" \
      -d "{\"input\":\"Timed request ${n} for Fish Audio S2 Pro.\",\"language\":\"english\"}" \
      --output "baseline-${n}.wav"
done

docker compose -f compose.yaml logs fish-sglang --tail 500
```

Record whether all three requests succeed and save the timings for comparison with the experiment.

- [ ] **Step 4: If baseline requests fail, stop and diagnose before touching CUDA graphs**

If the warmup request, any measured request, or the container logs show failures while `disable_cuda_graph=True`, stop here and diagnose before attempting the CUDA-graph experiment.

---

## Chunk 3: Run a Gated CUDA-Graph Experiment

### Task 3: Remove the explicit override only after baseline validation passes

**Files:**
- Modify: `Dockerfile.sglang` consolidated patch block only

- [ ] **Step 1: Change only the final CUDA-graph patch block**

After Chunk 1 and Chunk 2 pass, replace the final `stages.py` mutation in the consolidated RUN block with a verification-only check:

```dockerfile
# CUDA graphs experiment: stop overriding the upstream default, but fail fast
# if the pinned ref no longer defaults to disable_cuda_graph=False.
stages = Path('/opt/venv/lib/python3.12/site-packages/sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py')
text = stages.read_text(encoding='utf-8')
expected = '        disable_cuda_graph=False,\n'
if expected not in text:
    raise SystemExit('Expected disable_cuda_graph=False default not found in stages.py')
```

- [ ] **Step 2: Rebuild the experiment image**

```bash
docker compose -f compose.yaml build fish-sglang
docker compose -f compose.yaml up -d --force-recreate fish-sglang
```

If the verification fails here, treat the plan as stale relative to the pinned ref and inspect that ref before continuing.

- [ ] **Step 3: Repeat the same warmup and three measured requests**

Run the same command sequence used for the baseline so the timings are comparable.

- [ ] **Step 4: Accept the experiment only if all validation criteria pass**

Required acceptance criteria:
- `/v1/models` becomes healthy
- The warmup request succeeds
- All three measured speech requests succeed
- `fish-sglang` logs show no CUDA-graph startup or decode failures
- Average latency across the three measured requests is no worse than baseline by more than 5%

- [ ] **Step 5: If any criterion fails, revert only the CUDA-graph experiment**

If the container fails to start, request handling fails, or logs show CUDA-graph issues, restore the explicit disable patch:
```dockerfile
# Restore the compatibility override in the consolidated RUN block:
stages = Path('/opt/venv/lib/python3.12/site-packages/sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py')
text = stages.read_text(encoding='utf-8')
old = '        disable_cuda_graph=False,\n'
new = '        disable_cuda_graph=True,\n'
if old not in text:
    raise SystemExit('Expected disable_cuda_graph=False default not found in stages.py')
stages.write_text(text.replace(old, new, 1), encoding='utf-8')
```

Rebuild, confirm the baseline behavior is restored, and do not commit the experiment.

```bash
docker compose -f compose.yaml build fish-sglang
docker compose -f compose.yaml up -d --force-recreate fish-sglang
```

- [ ] **Step 6: Commit only after the experiment passes**

```bash
# Stage only the Dockerfile change; do not include unrelated untracked files.
git add Dockerfile.sglang
git commit -m "perf: re-enable CUDA graphs for Fish Audio"
```

- [ ] **Step 7: If the experiment fails, keep the cleanup-only commit as the final state**

That leaves the repo with fewer Docker layers and the existing compatibility behavior unchanged.

---

## Verification Commands

After completing the applicable chunk, run these checks:

Use PowerShell on the Windows host:

```powershell
# Build the service image
docker compose -f compose.yaml build fish-sglang

# Start the Fish Audio service from compose
docker compose -f compose.yaml up -d --build fish-sglang

# Verify the API is ready before speech requests
$ready = $false
for ($i = 1; $i -le 60; $i++) {
  try {
    curl.exe --fail --show-error "http://localhost:18001/v1/models" | Out-Null
    $ready = $true
    break
  }
  catch {
    Start-Sleep -Seconds 10
  }
}
if (-not $ready) {
  throw "fish-sglang did not become ready within 10 minutes"
}

# Check logs for startup and decode issues
docker compose -f compose.yaml logs fish-sglang --tail 500
```

Linux/bash equivalent:

```bash
# Build the service image
docker compose -f compose.yaml build fish-sglang

# Start the Fish Audio service from compose
docker compose -f compose.yaml up -d --build fish-sglang

# Verify the API is ready before speech requests
ready=0
for i in {1..60}; do
  if curl --fail --show-error "http://localhost:18001/v1/models"; then
    ready=1
    break
  fi
  sleep 10
done
if [ "$ready" -ne 1 ]; then
  echo "fish-sglang did not become ready within 10 minutes" >&2
  exit 1
fi

# Check logs for startup and decode issues
docker compose -f compose.yaml logs fish-sglang --tail 500
```

For timing comparisons, use either the Windows PowerShell or Linux/bash command sequence from Chunk 2 and repeat it unchanged for Chunk 3.

---

## Plan Summary

| Chunk | Tasks | Priority |
|-------|-------|----------|
| 1 | 1 | Consolidate patch layers and preserve current behavior |
| 2 | 2 | Collect a stable baseline with CUDA graphs disabled |
| 3 | 3 | Run a gated CUDA-graph experiment and commit only if it passes |

**Total: 3 tasks**
