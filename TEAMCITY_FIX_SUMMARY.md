# TeamCity PyTorch Fix — Strategy

## What

Make the PyTorch backend (`Gempy_TestingEnginePyTorch`) fully green on TeamCity.
It currently has 5 failures — all backend-agnostic code that assumes NumPy types and breaks
when `BackendTensor` is set to `PYTORCH`.

## How TeamCity is used

1. **`teamcity job list`** → find the right job (`Gempy_TestingEnginePyTorch`)
2. **`teamcity run start <job> --personal --branch @this --local-changes --watch`** →
   upload local changes as a **personal build** (doesn't pollute the main pipeline),
   waits for completion.
3. **`teamcity run tests <id> --failed`** / **`teamcity run log <id> --raw`** →
   extract failure details from the build log.
4. Iterate: fix locally → push to same branch → re-trigger personal build.

Command used:
```bash
teamcity run start Gempy_TestingEnginePyTorch \
  --personal --branch @this --local-changes --watch --timeout 30m \
  -m "Fix PyTorch tensor reduction"
```

## Failures being fixed

| # | Test | Error | Fix |
|---|------|-------|-----|
| 1 | `test_final_block_octrees` | `Tensor.astype()` doesn't exist | guard with `hasattr(..., 'cpu')` → `.cpu().numpy()` |
| 2-4 | `test_activator_*` (×3) | `repeat_interleave(numpy, Tensor, dim=…)` wrong types | convert args to Tensor in `BackendTensor._repeat` |
| 5 | `test_final_exported_fields_one_layer` | `reshape` 4050 → (15,2,15) shape mismatch | investigating |

## Files changed

- `gempy_engine/core/backend_tensor.py` — `_repeat` now auto-converts numpy→Tensor
- `tests/test_common/test_integrations/test_multi_fields.py` — Tensor-safe `.astype` call