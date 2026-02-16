import numpy as np
import matplotlib.pyplot as plt
from gempy_engine.modules.faults.finite_faults import (
    get_local_frame,
    get_ellipsoid_distance,
    spline_taper,
    cubic_hermite_taper
)

def test_visualize_uv_map_configurations():
    # 1. Define a 2D grid in local UV space
    # Since d = sqrt((u/a)^2 + (v/b)^2), we can just work in local coordinates directly
    u_range = np.linspace(-2.0, 2.0, 100)
    v_range = np.linspace(-2.0, 2.0, 100)
    U, V = np.meshgrid(u_range, v_range)
    
    # We'll simulate points as if they were already projected and centered
    # points_local = [U.ravel(), V.ravel(), 0]
    points_uv = np.stack([U.ravel(), V.ravel()], axis=-1)
    
    # Define some spline configurations
    cp_bell = np.array([
        [0.0, 1.0],
        [0.2, 0.95],
        [0.5, 0.5],
        [0.8, 0.05],
        [1.0, 0.0]
    ])
    
    cp_plateau = np.array([
        [0.0, 1.0],
        [0.6, 1.0],
        [0.8, 0.5],
        [1.0, 0.0]
    ])

    configurations = [
        {"name": "Isotropic Cubic (a=1, b=1)", "a": 1.0, "b": 1.0, "taper": "cubic"},
        {"name": "Anisotropic Strike (a=1.5, b=0.5)", "a": 1.5, "b": 0.5, "taper": "cubic"},
        {"name": "4-Direction Smoothness", "a": (1.5, 0.5), "b": (1.0, 0.5), "taper": "cubic"},
        {"name": "Rotated 45°", "a": 1.5, "b": 0.5, "taper": "cubic", "angle": 45},
        {"name": "Plateau Spline (a=1.2, b=0.8)", "a": 1.2, "b": 0.8, "taper": "spline_plateau"},
        {"name": "Rotated Spline 30°", "a": 1.5, "b": 0.8, "taper": "spline_bell", "angle": 30},
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, config in enumerate(configurations):
        a, b = config["a"], config["b"]
        
        # Apply rotation if specified
        angle_deg = config.get("angle", 0)
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Rotate grid
        U_rot = U * cos_a - V * sin_a
        V_rot = U * sin_a + V * cos_a
        
        # Calculate distance d using the updated get_ellipsoid_distance logic
        # For the sake of this test, we replicate the logic locally for U, V
        def local_distance(u, v, a, b):
            def get_r(val, radius):
                if isinstance(radius, (tuple, list)):
                    r_pos, r_neg = radius
                    return np.where(val >= 0, r_pos, r_neg)
                return radius
            a_mask = get_r(u, a)
            b_mask = get_r(v, b)
            return np.sqrt((u / a_mask) ** 2 + (v / b_mask) ** 2)

        d = local_distance(U_rot, V_rot, a, b)
        
        # Apply taper
        if config["taper"] == "cubic":
            slip = cubic_hermite_taper(d)
        elif config["taper"] == "spline_bell":
            slip = spline_taper(d.ravel(), cp_bell).reshape(d.shape)
        elif config["taper"] == "spline_plateau":
            slip = spline_taper(d.ravel(), cp_plateau).reshape(d.shape)
        
        im = axes[i].imshow(
            slip, 
            extent=[u_range[0], u_range[-1], v_range[0], v_range[-1]],
            origin='lower',
            cmap='viridis',
            vmin=0, vmax=1
        )
        axes[i].set_title(config["name"])
        axes[i].set_xlabel("Strike (u)")
        axes[i].set_ylabel("Dip (v)")
        
        # Add a contour at d=1 (the tip line)
        axes[i].contour(U_rot, V_rot, d, levels=[1.0], colors='red', linestyles='dashed')

    fig.colorbar(im, ax=axes.tolist(), label='Slip Multiplier')
    plt.suptitle("Finite Fault UV Slip Maps - Various Anisotropy and Tapers", fontsize=16)
    
    plt.savefig('tests/test_common/test_api/test_faults/finite_fault_uv_maps.png')
    plt.show()

    print("UV map configurations plotted.")

def test_axis_definitions_explanation():
    """
    This test verifies the get_local_frame logic and prints out an explanation 
    of how axes are defined, as requested by the user.
    """
    # Example normal vector (dipping 45 degrees East)
    # Normal is [ -sin(45), 0, cos(45) ] roughly if dip is 45 East (90 strike)
    # Actually let's just pick a vector.
    normal = np.array([1.0, 0.0, 1.0]) # 45 degree dip to the West (if Z is up, X is East)
    u, v, w = get_local_frame(normal)
    
    print(f"\nAxis Definitions for Normal {normal}:")
    print(f"Normal (w): {w} - Direction perpendicular to the fault plane.")
    print(f"Strike (u): {u} - Horizontal vector on the fault plane. (w x Z)")
    print(f"Dip (v):    {v} - Vector pointing down-dip on the fault plane. (u x w)")
    
    # Assertions for orthogonality
    assert np.isclose(np.dot(u, v), 0.0)
    assert np.isclose(np.dot(u, w), 0.0)
    assert np.isclose(np.dot(v, w), 0.0)
    assert np.isclose(u[2], 0.0) # Strike should be horizontal (Z component = 0)

if __name__ == "__main__":
    test_visualize_uv_map_configurations()
    test_axis_definitions_explanation()
