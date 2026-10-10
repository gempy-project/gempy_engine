# Field Evaluation After `compute_model`

`prepare_field_evaluator` / `evaluate_fields` evaluate a computed model's raw
stack scalar fields at arbitrary points, from the weights the production solve
kept on its outputs. Nothing is solved again and the weight cache is not used
(it is already cleared when `compute_model` returns). It is the engine side of
volume-first meshing: a cutter in `gempy_plugins` evaluates the fields on its
own background mesh, so no dual-contouring surfaces are needed.

```python
from gempy_engine import compute_model, prepare_field_evaluator, classify_units

solutions = compute_model(interpolation_input, options, data_descriptor)
evaluator = prepare_field_evaluator(solutions, interpolation_input, data_descriptor, options)

values = evaluator(points)                        # (S, M), production fault drift
faults = evaluator.fault_values(points)           # (F, M), fault scalar minus its level
sides = np.where(faults > 0, 1, -1)               # or the side of each point's fault block
blocks = evaluator(points, fault_sides=sides)     # each fault's drift fixed to that side
stack, interval = classify_units(values, evaluator.stack_relations, evaluator.stack_isovalues)
```

- `evaluator(points, stacks=None, fault_sides=None)` returns raw scalars, rows
  following `stacks` (all by default). Without `fault_sides` the affected
  stacks see the production drift (each fault's soft-segmented scalar), so the
  values equal the production fields on the octree points.
- `fault_sides` is `(len(evaluator.fault_stacks), M)` of `+1` / `-1`: affected
  stacks see the drift plateau of that side instead of the soft step. A fault
  block's field is then smooth up to and across its faults; evaluating a point
  with both sides gives the throw there.
- `classify_units(values, relations, isovalues)` is the hard version of the
  production lithology masking (erosion, onlap chains, preceding-mask
  exclusion): active stack and interval per point. `unit_ids` maps them to the
  production unit ids.
- `evaluate_fields(...)` is the one-shot form. Prepare once for repeated
  evaluation: preparation freezes solver inputs and verifies the drift rule on
  the full reference grid.

Pass the same input, descriptor and options as to `compute_model`. Supported:
NumPy or Torch (CPU or CUDA, optionally KeOps), float64, no autograd;
independent infinite faults; no micro corrections, external interpolation or
segmentation functions, or null-space stacks. With KeOps, stacks with equal
kernel options are evaluated in stacked block-sparse reductions.

Weisweiler (12 stacks, 10 faults, KeOps GPU): preparation 0.4 s, 20 000 points
with fixed sides 0.7 s.

Source: `API/model/field_evaluation.py` (`FieldEvaluator`), the sampler's
`query_sides` in `API/dual_contouring/fault_drift_sampler.py`, and
`modules/activator/unit_classification.py`. Tests:
`tests/test_common/test_modules/test_field_evaluation.py`.
