### Context

Finite faults are implemented by combining the scalar field generated 
 by the fault input data with another implicit field between 0 and 1 describing
 the offset where 1 is maximal offset. The sum of these two scalar fields is used
 as a drift function for the following surfaces.

The best way to define the offset scalar field or at least the most intuitive for a 
geologist would be to paint a uv map on the fault surface. Then we would need to project
this uv map into the full 3D space.

My current best idea for this is to project all the points of the full 3D space (grid centers, etc.)
into the fault surface and then use the UV map defined on that surface to get the offset scalar field.
To project the points we use the gradient of the scalar field of the fault surface.

### Detailed Phased Plan

#### Phase 1: Local Coordinate System & Analytical Ellipsoid UV
**Goal**: Establish a local coordinate system (u, v, w) on the fault and define an analytical UV mapping using an ellipsoid.
- **Task 1.1**: Implement a function to compute a local orthonormal basis (Strike, Dip, Normal) given a gradient vector and a center point.
- **Task 1.2**: Implement analytical ellipsoid UV mapping: given a projected point $P'$, calculate its normalized distance $d$ from the center in the local (u, v) plane.
- **Tests**:
    - `test_local_frame_orthogonal`: Verify that the generated basis is orthonormal.
    - `test_ellipsoid_distance`: Verify that points on the ellipsoid boundary have $d=1$, inside $d<1$, and outside $d>1$.

#### Phase 2: Point Projection ("Walking the Gradient")
**Goal**: Project any 3D point $P$ onto the nearest point $P'$ on the fault surface $F(x,y,z)=c$.
- **Task 2.1**: Implement the projection formula: $P' = P - 0.5 \cdot (F(P) - c) \frac{\nabla F(P)}{\|\nabla F(P)\|^2}$. 
    - *Note*: We use a 0.5 factor because GemPy scalar fields typically behave quadratically ($F \approx d^2$) near the fault surface, so the gradient is twice as strong as a standard SDF gradient.
- **Task 2.2**: Handle potential instabilities where $\|\nabla F\|$ is near zero.
- **Tests**:
    - `test_projection_on_plane`: Verify projection works perfectly for a simple tilted plane.
    - `test_projection_consistency`: Verify that projecting $P'$ again results in $P'$ (idempotency).

#### Phase 3: Slip Tapering Functions
**Goal**: Convert the normalized distance $d$ into a slip multiplier [0, 1].
- **Task 3.1**: Implement Cubic Hermite (Smoothstep) taper: $S(d) = 1 - (3d^2 - 2d^3)$ for $d < 1$, else 0.
- **Task 3.2**: Implement Quadratic taper: $S(d) = (1 - d^2)^2$ for $d < 1$, else 0.
- **Tests**:
    - `test_taper_bounds`: Verify $S(0)=1$ and $S(1)=0$.
    - `test_taper_smoothness`: Verify derivatives at boundaries if applicable.

#### Phase 4: Integration and Visualization
**Goal**: Combine all steps into a finite fault offset field and visualize.
- **Task 4.1**: Create a function that takes a GemPy solution, a fault index, and ellipsoid parameters, and returns the 3D offset field.
- **Task 4.2**: Integrate with PyVista to show the UV coordinates and the resulting slip on the fault mesh.
- **Tests**:
    - `test_finite_fault_full_pipeline`: A high-level test that runs GemPy, applies the UV projection, and checks if the offset field looks correct.

### Step 1: UV implicit ellipsoid gradient
The Analytical Slip Pipeline. To make this work, you essentially need to define a local 2D coordinate system on your fault plane, calculate how far your projected point is from the center, and feed that distance into a spline.

1. Defining the Strike and Dip Axes (Local Frame)
Before you can evaluate an ellipsoid, you need to orient it. You can define an orthogonal basis $(u, v, w)$ at the center point of your fault $C$.
Normal ($w$): This is the gradient of your fault field at the center: $w = \frac{\nabla F(C)}{\|\nabla F(C)\|}$.
Strike ($u$): Strike is typically the horizontal line across the fault surface. You can find this by taking the cross product of the normal vector and the global vertical axis (usually the Z-axis, $[0, 0, 1]$):
$$u = \frac{w \times [0, 0, 1]}{\|w \times [0, 0, 1]\|}$$
Dip ($v$): The dip vector points down the slope of the fault. You get this via the cross product of strike and the normal:
$$v = u \times w$$

2. The Implicit Ellipsoid (Normalized Distance)
Once your 3D point $P$ is projected onto the fault surface as $P'$, you need to find its local coordinates relative to the fault center $C$.
You project the vector $(P' - C)$ onto your Strike and Dip axes using the dot product:
$x_{local} = (P' - C) \cdot u$
$y_{local} = (P' - C) \cdot v$
Now, you evaluate the implicit equation of an ellipse. Let $a$ be the fault radius along the strike, and $b$ be the fault radius along the dip. We can calculate a normalized distance $d$:
$$d = \sqrt{\left(\frac{x_{local}}{a}\right)^2 + \left(\frac{y_{local}}{b}\right)^2}$$
If $d \ge 1$, the point is outside the fault tip line. Offset = $0$.
If $d = 0$, the point is exactly at the center of the fault. Offset = Max Slip.

3. The Smoothing Function (Splines)
Your intuition to use splines is exactly right. Because you have a normalized distance $d$ ranging from $0$ to $1$, you just need a 1D taper function to map that distance to an offset value.
Geological slip profiles are often modeled using bell curves or polynomial falloffs so that the displacement gracefully reaches zero at the tips without causing sharp kinks in the rock layers.

Here are three excellent smoothing options:
- The Cubic Hermite (Smoothstep): The classic computer graphics smoothing function. It provides zero derivatives at both ends ($d=0$ and $d=1$), ensuring the rock layers bend perfectly smoothly into the fault.
$$Offset(d) = MaxSlip \times (1 - d^2 \cdot (3 - 2d))$$
- The Quadratic Taper: A slightly sharper, very standard geological profile.
$$Offset(d) = MaxSlip \times (1 - d^2)^2$$
- 1D B-Spline: If you want the geologist to have ultimate control, you can expose a UI with a 1D cubic spline curve mapping $d$ on the X-axis to an offset multiplier on the Y-axis.

### Step 2: Visualization of the uv coordinates onto a 3D plane

### Step 3: Grid and other input points projection onto the fault surface
1. The Projection Mechanism (Walking the Gradient)
Your idea to interpolate the gradient of the fault's scalar field $F(x, y, z)$ is exactly the right path. The gradient $\nabla F$ acts as a vector field pointing perpendicularly toward/away from the fault surface.
The Math: If your fault scalar field is a true (or approximate) Signed Distance Function (SDF), projecting a 3D point $P$ to its corresponding point on the fault surface $P'$ requires a single mathematical step:
$$P' = P - F(P) \frac{\nabla F(P)}{\|\nabla F(P)\|}$$
Note: If $F$ is not an SDF, a better approximation is $P' = P - 0.5 \cdot F(P) \frac{\nabla F(P)}{\|\nabla F(P)\|^2}$. The 0.5 factor accounts for quadratic behavior of GemPy's scalar field ($F \approx d^2$) where $\nabla F \approx 2d$.