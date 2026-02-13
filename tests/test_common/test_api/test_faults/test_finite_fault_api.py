
import numpy as np
import pytest
from gempy_engine.modules.faults.finite_faults import FiniteFault, TaperType

def test_finite_fault_api_basic():
    center = np.array([0, 0, 0])
    ff = FiniteFault(
        center=center,
        strike_radius=2.0,
        dip_radius=1.0,
        taper=TaperType.CUBIC
    )
    
    # Points on strike/dip boundaries
    points = np.array([
        [0, 0, 0],
        [2, 0, 0], # +u boundary (u=[1,0,0])
        [0, 0, 1],  # +v boundary (v=[0,0,1])
        [3, 0, 0]
    ])
    
    # Simple vertical fault (normal along Y)
    # w=[0,1,0], u=[1,0,0], v=[0,0,1]
    normal = np.array([0, 1.0, 0])
    
    slip = ff.calculate_slip(points, normal)
    
    assert np.isclose(slip[0], 1.0) # Center
    assert np.isclose(slip[1], 0.0) # Strike boundary
    assert np.isclose(slip[2], 0.0) # Dip boundary
    assert np.isclose(slip[3], 0.0) # Outside

def test_finite_fault_anisotropy():
    center = np.array([0, 0, 0])
    ff = FiniteFault(
        center=center,
        strike_radius=(2.0, 1.0), # 2.0 in +u, 1.0 in -u
        dip_radius=1.0
    )
    
    points = np.array([
        [2, 0, 0],   # +u boundary (u=[1,0,0], Radius_pos=2)
        [-1, 0, 0],  # -u boundary (u=[1,0,0], Radius_neg=1)
        [1, 0, 0],   # inside +u (x_local=1. Radius_pos=2)
        [-0.5, 0, 0] # inside -u (x_local=-0.5. Radius_neg=1)
    ])
    
    normal = np.array([0, 1.0, 0]) # w=[0,1,0], u=[1,0,0], v=[0,0,1]
    slip = ff.calculate_slip(points, normal)
    
    assert np.isclose(slip[0], 0.0)
    assert np.isclose(slip[1], 0.0)
    assert 0 < slip[2] < 1.0
    assert 0 < slip[3] < 1.0

def test_finite_fault_spline():
    center = np.array([0, 0, 0])
    cp_custom = np.array([
        [0.0, 1.0],
        [0.5, 1.0], # Plateau
        [1.0, 0.0]
    ])
    ff = FiniteFault(
        center=center,
        taper=TaperType.SPLINE,
        spline_control_points=cp_custom
    )
    
    points = np.array([
        [0, 0, 0],
        [0.4, 0, 0] # d=0.4, should be on plateau (near 1.0)
    ])
    normal = np.array([0, 0, 1.0])
    slip = ff.calculate_slip(points, normal)
    
    assert np.isclose(slip[0], 1.0)
    assert slip[1] > 0.9

if __name__ == "__main__":
    pytest.main([__file__])
