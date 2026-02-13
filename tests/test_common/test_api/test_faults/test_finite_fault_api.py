
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

def test_finite_fault_rotation():
    center = np.array([0, 0, 0])
    # Anisotropic fault: long axis (2.0) along strike, short (1.0) along dip
    ff = FiniteFault(
        center=center,
        strike_radius=2.0,
        dip_radius=1.0,
        rotation=90 # Rotate 90 deg
    )
    
    # Normal along Z => default strike u=[1,0,0], dip v=[0,1,0]
    # In get_local_frame, if w is [0,0,1], u = w x [1,0,0] = [0,1,0]
    # Then rotate u by 90 deg around w:
    # u_rot = [0,1,0]*cos(90) + [0,0,1]x[0,1,0]*sin(90) = [-1,0,0]
    # v_rot = w x u_rot = [0,0,1] x [-1,0,0] = [0,-1,0]
    normal = np.array([0, 0, 1.0])
    
    points = np.array([
        [-2, 0, 0],  # On the new strike boundary (u_rot=[-1,0,0], a=2.0) => d=1.0
        [0, -1, 0],  # On the new dip boundary (v_rot=[0,-1,0], b=1.0) => d=1.0
    ])
    
    slip = ff.calculate_slip(points, normal)
    
    # We use small tolerance as spline interpolation or float precision might vary slightly from exactly 0
    assert np.allclose(slip, 0.0, atol=1e-12)

if __name__ == "__main__":
    pytest.main([__file__])
