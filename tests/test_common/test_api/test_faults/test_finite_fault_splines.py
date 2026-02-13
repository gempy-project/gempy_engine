import numpy as np
import matplotlib.pyplot as plt
from gempy_engine.modules.faults.finite_faults import spline_taper, cubic_hermite_taper, quadratic_taper

def test_spline_taper_configurations():
    # Define distance range for plotting
    d = np.linspace(0, 1.2, 200)

    # 1. Linear-like spline (just two points)
    # Note: cubic spline needs at least 4 points for k=3 by default in some versions,
    # or it might fall back. Let's provide 4 points for safety.
    cp_linear = np.array([
        [0.0, 1.0],
        [0.33, 0.67],
        [0.67, 0.33],
        [1.0, 0.0]
    ])
    
    # 2. Bell-shaped spline
    cp_bell = np.array([
        [0.0, 1.0],
        [0.2, 0.95],
        [0.5, 0.5],
        [0.8, 0.05],
        [1.0, 0.0]
    ])

    # 3. Custom complex spline (e.g., constant slip then sharp drop)
    cp_plateau = np.array([
        [0.0, 1.0],
        [0.5, 1.0],
        [0.7, 0.8],
        [0.9, 0.2],
        [1.0, 0.0]
    ])

    # 4. Spline with an increase (geologically weird but possible for testing)
    cp_weird = np.array([
        [0.0, 0.5],
        [0.3, 1.0],
        [0.7, 0.2],
        [1.0, 0.0]
    ])

    taper_linear = spline_taper(d, cp_linear)
    taper_bell = spline_taper(d, cp_bell)
    taper_plateau = spline_taper(d, cp_plateau)
    taper_weird = spline_taper(d, cp_weird)

    # Standard tapers for comparison
    taper_cubic = cubic_hermite_taper(d)
    taper_quad = quadratic_taper(d)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(d, taper_cubic, '--', label='Cubic Hermite (Standard)', alpha=0.5)
    plt.plot(d, taper_quad, '--', label='Quadratic (Standard)', alpha=0.5)
    
    plt.plot(d, taper_linear, label='Spline: Linear-like')
    plt.plot(d, taper_bell, label='Spline: Bell-shaped')
    plt.plot(d, taper_plateau, label='Spline: Plateau')
    plt.plot(d, taper_weird, label='Spline: Weird')

    plt.axvline(1.0, color='k', linestyle=':', label='Fault Tip (d=1)')
    plt.title('Finite Fault Slip Tapering - Spline Configurations')
    plt.xlabel('Normalized Distance (d)')
    plt.ylabel('Slip Multiplier')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('tests/test_common/test_api/test_faults/spline_taper_plots.png')
    plt.show()

    # Assertions
    assert np.isclose(taper_linear[0], 1.0)
    assert np.isclose(taper_linear[-1], 0.0)
    assert taper_plateau[int(len(d)*0.4)] > 0.9 # Should be near 1.0 at d=0.48
    
    print("Spline taper configurations tested and plotted.")

if __name__ == "__main__":
    test_spline_taper_configurations()
