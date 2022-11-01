"""
The "hello, world" of computational PDEs:  Solve the Poisson equation,
verifying accuracy via the method of manufactured solutions.

This example uses the simplest IGA discretization, namely, explicit B-splines
in which parametric and physical space are the same.
"""

import tIGAr
from tIGAr import BSplines
import math

# Number of levels of refinement with which to run the Poisson problem.
# (Note: Paraview output files will correspond to the last/highest level
# of refinement.)
N_LEVELS = 6

# Array to store error at different refinement levels:
L2_errors = []


def poisson(level: int):

    # Preprocessing
    # Parameters determining the polynomial degree and number of elements in
    # each parametric direction.  By changing these and recording the error,
    # it is easy to see that the discrete solutions converge at optimal rates
    # under refinement.
    p = 2
    NELu = (2**level)
    mpirank = tIGAr.mpirank
    # Parameters determining the position and size of the domain.
    x0, Lx = 0.0, 1.0

    if mpirank == 0:
        print("Generating extraction...")

    # Create a control mesh for which $\Omega = \widehat{\Omega}$.
    splineMesh = BSplines.ExplicitBSplineControlMesh(
        [p], [BSplines.uniformKnots(p, x0, x0+Lx, NELu)])

    # Create a spline generator for a spline with a single scalar field on the
    # given control mesh, where the scalar field is the same as the one used
    # to determine the mapping $\mathbf{F}:\widehat{\Omega}\to\Omega$.
    splineGenerator = BSplines.EqualOrderSpline(1, splineMesh)

    # Set Dirichlet boundary conditions on the 0-th (and only) field, on both
    # ends of the domain, in both directions.
    field = 0
    scalarSpline = splineGenerator.getScalarSpline(field)
    for side in [0, 1]:
        sideDofs = scalarSpline.getSideDofs(0, side)
        splineGenerator.addZeroDofs(field, sideDofs)
    # Alternative: set BCs based on location of corresponding control points.
    # (Note that this only makes sense for splineGenerator of type
    # EqualOrderSpline; for non-equal-order splines, there is not
    # a one-to-one correspondence between degrees of freedom and geometry
    # control points.)

    # Write extraction data to the filesystem.
    DIR = "./extraction"
    splineGenerator.writeExtraction(DIR)

    # Analysis

    if mpirank == 0:
        print("Setting up extracted spline...")

    # Choose the quadrature degree to be used throughout the analysis.
    # In IGA, especially with rational spline spaces, under-integration is a
    # fact of life, but this does not impair optimal convergence.
    QUAD_DEG = 2*p

    # Create the extracted spline directly from the generator.
    # As of version 2019.1, this is required for using quad/hex elements in
    # parallel.
    spline = BSplines.ExtractedSpline(splineGenerator, QUAD_DEG)

    # Alternative: Can read the extracted spline back in from the filesystem.
    # For quad/hex elements, in version 2019.1, this only works in serial.

    if mpirank == 0:
        print("Solving...")

    # Homogeneous coordinate representation of the trial function u.  Because
    # weights are 1 in the B-spline case, this can be used directly in the PDE,
    # without dividing through by weight.
    u = tIGAr.TrialFunction(spline.V)
    # u = tIGAr.Function(spline.V)
    # Corresponding test function.
    v = tIGAr.TestFunction(spline.V)

    # Create a force, f, to manufacture the solution, soln
    x = spline.spatialCoordinates()
    sin = tIGAr.sin
    pi = tIGAr.pi
    soln = sin(pi*(x[0]-x0)/Lx)
    f = -spline.div(spline.grad(soln))

    # Set up and solve the Poisson problem
    a = tIGAr.inner(spline.grad(u), spline.grad(v))*spline.dx
    L = tIGAr.inner(f, v)*spline.dx
    # u = tIGAr.Function(spline.V)
    M, b = spline.assembleLinearSystem(a, -L, False)
    print(b[:])
    return
    spline.solveLinearVariationalProblem(a == L, u)
    return
    # Postprocessing #######

    # The solution, u, is in the homogeneous representation, but, again, for
    # B-splines with weight=1, this is the same as the physical representation.
    # File("results/u.pvd") << u

    # Compute and print the $L^2$ error in the discrete solution.
    L2_error = math.sqrt(tIGAr.assemble(((u-soln)**2)*spline.dx))
    L2_errors.append(L2_error)
    if level > 0:
        rate = math.log(L2_errors[level-1]/L2_errors[level])/math.log(2.0)
    else:
        rate = "--"
    if mpirank == 0:
        print("L2 Error for level "+str(level)+" = "+str(L2_error)
              + "  (rate = "+str(rate)+")")


if __name__ == "__main__":
    # for level in range(N_LEVELS):
    #    poisson(level)
    poisson(2)
