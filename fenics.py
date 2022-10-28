import numpy as np
from dolfin import *
import matplotlib.pyplot as plt

dt = 1e-3
mesh = IntervalMesh(8000, 0, 1)

element = FiniteElement('CG', mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, element)

rhoc = 0.5
am = 0.5 * (3.0 - rhoc) / (1.0 + rhoc)
af = 1.0 / (1.0 + rhoc)
gamma = 0.5 + am -af

v = TestFunction(W)
dT1 = Function(W)
dT0 = Function(W)
T0 = Function(W)

dTm = am * dT1 + (1 - am) * dT0
Tm = T0 + af * dt * (gamma * dT1 + (1 - gamma) * dT0)

F = dot(v, dTm) * dx + dot(grad(v), grad(Tm)) * dx

class Init(UserExpression):
    def eval(self, values, x):
        values[0] = np.sin(np.pi * x)
    def value_shape(self):
        return ()

if __name__ == '__main__':
    T0.interpolate(Init())
    for i in range(50):
        dT1.vector()[:] = (gamma - 1) / gamma
        solve(F == 0, dT1)
        T0.vector()[:] += dt * (gamma * dT1.vector() + (1 - gamma) * dT0.vector())

    y = T0.compute_vertex_values().reshape(-1, 1)
    x = mesh.coordinates()
    plt.plot(x, y)
    plt.show()
    
    np.savetxt('ref.txt', np.hstack((x,y)))
