import pytest

from gempy_engine.API.dual_contouring import joint_extraction
from gempy_engine.API.dual_contouring.multi_scalar_dual_contouring import dual_contouring_multi_scalar
from gempy_engine.config import DualContouringOverlap
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.exceptions import MeshExtractionError

JUNCTION = "unsupported_sampled_junction: multiple tile roots"


@pytest.mark.parametrize("message, code", [
    (JUNCTION, "unsupported_sampled_junction"),
    ("degenerate_sampled_face: tangential intersection", "degenerate_sampled_face"),
    ("nonfinite_qef_solution: original production solve failed", "nonfinite_qef_solution"),
])
def test_coded_value_errors_become_mesh_extraction_errors(message, code):
    error = MeshExtractionError.from_error(ValueError(message))
    assert isinstance(error, ValueError)
    assert (str(error), error.code, error.reason) == (message, code, "mesh_extraction_failed")
    assert error.context == {"code": code}


@pytest.mark.parametrize("error", [
    ValueError("Octree grid must be defined to extract the mesh"),
    ValueError("plain: no underscore in the code"),
    RuntimeError(JUNCTION),
])
def test_other_errors_are_not_mesh_extraction_errors(error):
    assert MeshExtractionError.from_error(error) is None


@pytest.mark.parametrize("mode, name", [
    (DualContouringOverlap.joint_contacts, "extract_joint_contacts_octree"),
    (DualContouringOverlap.joint, "extract_joint_octree"),
])
def test_joint_dispatch_raises_the_code(monkeypatch, mode, name):
    def fail(*args):
        raise ValueError(JUNCTION)

    monkeypatch.setattr(joint_extraction, name, fail)
    options = InterpolationOptions.from_args(range=1.0, c_o=1.0, number_octree_levels=1)
    options.evaluation_options.mesh_extraction_overlap = mode

    with pytest.raises(MeshExtractionError, match=JUNCTION) as raised:
        dual_contouring_multi_scalar(None, None, options, [])
    assert raised.value.code == "unsupported_sampled_junction"
    assert isinstance(raised.value.__cause__, ValueError)


def test_joint_dispatch_keeps_uncoded_errors(monkeypatch):
    def fail(*args):
        raise ValueError("something else")

    monkeypatch.setattr(joint_extraction, "extract_joint_contacts_octree", fail)
    options = InterpolationOptions.from_args(range=1.0, c_o=1.0, number_octree_levels=1)
    options.evaluation_options.mesh_extraction_overlap = DualContouringOverlap.joint_contacts

    with pytest.raises(ValueError, match="something else") as raised:
        dual_contouring_multi_scalar(None, None, options, [])
    assert type(raised.value) is ValueError
