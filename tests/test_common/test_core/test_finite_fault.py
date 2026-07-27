import json

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from gempy_engine.core.data.finite_fault import FiniteFault, TaperType


def test_finite_fault_json_round_trip():
    finite_fault = FiniteFault(
        center=np.array([1.0, 2.0, 3.0]),
        strike_radius=(2.0, 1.0),
        dip_radius=0.75,
        taper=TaperType.SPLINE,
        rotation_deg=15.0,
        spline_control_points=np.array([
                [0.0, 1.0],
                [0.5, 0.8],
                [1.0, 0.0],
        ]),
    )
    adapter = TypeAdapter(FiniteFault)

    payload = adapter.dump_json(finite_fault)
    restored = adapter.validate_json(payload)

    assert restored == finite_fault
    assert json.loads(payload) == {
            "center"                : [1.0, 2.0, 3.0],
            "strike_radius"         : [2.0, 1.0],
            "dip_radius"            : 0.75,
            "taper"                 : "spline",
            "rotation_deg"          : 15.0,
            "spline_control_points": [[0.0, 1.0], [0.5, 0.8], [1.0, 0.0]],
    }


@pytest.mark.parametrize(
    "field,value",
    [
            ("strike_radius", 0.0),
            ("strike_radius", (1.0, -1.0)),
            ("dip_radius", float("inf")),
    ],
)
def test_finite_fault_rejects_invalid_radii(field, value):
    with pytest.raises(ValidationError, match="radii must be finite and greater than zero"):
        FiniteFault(center=(0.0, 0.0, 0.0), **{field: value})


@pytest.mark.parametrize(
    "points,error",
    [
            (((0.1, 1.0), (1.0, 0.0)), "must start at 0"),
            (((0.0, 1.0), (0.5, 0.8), (0.5, 0.4), (1.0, 0.0)), "strictly increasing"),
            (((0.0, 1.1), (1.0, 0.0)), "must be between 0 and 1"),
    ],
)
def test_finite_fault_rejects_invalid_spline_control_points(points, error):
    with pytest.raises(ValidationError, match=error):
        FiniteFault(
            center=(0.0, 0.0, 0.0),
            taper=TaperType.SPLINE,
            spline_control_points=points,
        )


def test_finite_fault_rejects_spline_points_for_polynomial_taper():
    with pytest.raises(ValidationError, match="require a spline taper"):
        FiniteFault(
            center=(0.0, 0.0, 0.0),
            spline_control_points=((0.0, 1.0), (1.0, 0.0)),
        )


def test_finite_fault_rejects_unknown_fields_on_deserialization():
    adapter = TypeAdapter(FiniteFault)

    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        adapter.validate_python({"center": [0.0, 0.0, 0.0], "unknown": True})
