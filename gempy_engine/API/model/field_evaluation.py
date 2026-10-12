"""Raw stack fields of a computed model at arbitrary points, from its fixed production weights.

For downstream meshers (e.g. a volume-first tetrahedral cutter) that evaluate
the model after ``compute_model`` returned: no new solve, no weight cache.
Optionally each fault's drift is fixed to one side per point (``fault_sides``),
so a fault block's field stays smooth up to and across its faults.
"""

from dataclasses import dataclass, field

import numpy as np

from ..dual_contouring.fault_drift_sampler import prepare_fault_drift_sampler
from ..dual_contouring.fixed_weight_snapshots import host
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.options import InterpolationOptions
from ...core.data.solutions import Solutions


@dataclass(frozen=True)
class FieldEvaluator:
    """Callable ``(points, stacks=None, fault_sides=None) -> (S, M)`` raw stack scalars.

    ``fault_stacks`` lists the (independent, infinite) fault stacks, the rows of
    ``fault_sides``; ``affected_by`` maps each stack to the faults that offset
    it; ``stack_isovalues`` are each stack's surface levels and
    ``stack_relations`` its masking relation, as ``classify_units`` expects.
    """
    fault_stacks: tuple
    affected_by: dict
    stack_isovalues: list
    stack_relations: list
    diagnostics: dict
    _sampler: dict = field(repr=False)

    def __call__(self, points, stacks=None, fault_sides=None) -> np.ndarray:
        """Production fault drift by default; with ``fault_sides`` (F, M) of ±1, each fault's plateau."""
        stacks = tuple(range(len(self.stack_relations))) if stacks is None else tuple(stacks)
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if fault_sides is None:
            values, = self._sampler['query_batch']([(points, stacks, 'scalar')])
            return values
        return self._sampler['query_sides'](points, stacks, fault_sides)

    def fault_values(self, points) -> np.ndarray:
        """(F, M) fault scalars minus their surface level: positive on the positive side."""
        if not self.fault_stacks:
            return np.empty((0, len(np.asarray(points).reshape(-1, 3))))
        levels = np.array([self.stack_isovalues[f][0] for f in self.fault_stacks])
        return self(points, stacks=self.fault_stacks) - levels[:, None]


def prepare_field_evaluator(solutions: Solutions, interpolation_input: InterpolationInput,
                            data_descriptor: InputDataDescriptor, options: InterpolationOptions) -> FieldEvaluator:
    """Freeze ``solutions``' production weights once for repeated evaluation.

    Pass the same input, descriptor and options as to ``compute_model``. NumPy
    or Torch (CPU or CUDA, optionally KeOps), float64, no autograd. Faults must
    be independent and infinite; micro corrections and external interpolation
    functions are not supported.
    """
    octree_list = solutions.octrees_output
    sampler = prepare_fault_drift_sampler(data_descriptor, interpolation_input, options, octree_list)
    stack_isovalues = [np.asarray(host(output.scalar_field_at_sp), dtype=np.float64).reshape(-1).copy()
                       for output in octree_list[0].outputs]
    return FieldEvaluator(fault_stacks=tuple(sampler['fault_stacks']), affected_by=dict(sampler['affected_by']),
                          stack_isovalues=stack_isovalues,
                          stack_relations=list(data_descriptor.stack_structure.masking_descriptor),
                          diagnostics=dict(sampler['diagnostics']), _sampler=sampler)


def evaluate_fields(solutions: Solutions, interpolation_input: InterpolationInput,
                    data_descriptor: InputDataDescriptor, options: InterpolationOptions, points,
                    stacks=None, fault_sides=None) -> np.ndarray:
    """One-shot ``prepare_field_evaluator(...)(points, stacks, fault_sides)``; (S, M) raw stack scalars."""
    return prepare_field_evaluator(solutions, interpolation_input, data_descriptor, options)(
        points, stacks=stacks, fault_sides=fault_sides)
