import sys
import numpy as np
from dolfin import *
import matplotlib.pyplot as plt

dt = 1e-3/16
n = 40
if len(sys.argv) == 2:
    n = int(sys.argv[1])
mesh = IntervalMesh(n, 0, 1)

element = FiniteElement('CG', mesh.ufl_cell(), 2)
W = FunctionSpace(mesh, element)

rhoc = 0.5
am = 0.5 * (3.0 - rhoc) / (1.0 + rhoc)
af = 1.0 / (1.0 + rhoc)
gamma = 0.5 + am -af

prob = 'hc' # 'hc' or 'ps' or 'l2'
if prob == 'hc':
    v = TestFunction(W)
    dT1 = Function(W)
    dT0 = Function(W)
    T0 = Function(W)

    dTm = am * dT1 + (1 - am) * dT0
    Tm = T0 + af * dt * (gamma * dT1 + (1 - gamma) * dT0)

    F = dot(v, dTm) * dx + dot(grad(v), grad(Tm)) * dx
elif prob == 'l2':
    v = TestFunction(W)
    T0 = Function(W)
    xx = SpatialCoordinate(mesh)
    F = dot(v, T0 - cos(pi * xx[0])) * dx

elif prob == 'ps':
    v = TestFunction(W)
    T0 = Function(W)
    xx = SpatialCoordinate(mesh)
    F = dot(grad(v), grad(T)) * dx + dot(v, -pi*pi*cos(pi * xx[0])) * dx
        
class Init(UserExpression):
    def eval(self, values, x):
        values[0] = np.cos(np.pi * x)
    def value_shape(self):
        return ()
class Init2(UserExpression):
    def eval(self, values, x):
        values[0] = -np.pi**2*np.cos(np.pi * x)
    def value_shape(self):
        return ()

if __name__ == '__main__':
    set_log_level(50)

    if prob == 'l2' or prob == 'ps':
        solve(F == 0, T0)
    elif prob == 'hc':
        T0.interpolate(Init())
        dT0.interpolate(Init2())
        dT1.interpolate(Init2())
        for i in range(80):
            dT1.vector()[:] = (gamma - 1) / gamma
            solve(F == 0, dT1)
            T0.vector()[:] += dt * (gamma * dT1.vector() + (1 - gamma) * dT0.vector())
            dT0.vector()[:] = dT1.vector()

    #x = mesh.coordinates().squeeze()
    x = np.linspace(0, 1, 1001)
    #y = T0.compute_vertex_values().squeeze()
    y = np.array([T0(xc) for xc in x])
    plt.plot(x, y)
    T = np.exp(-pi**2*0.005) * np.cos(pi*x)
    plt.plot(x, T)
    plt.show()
    
    pi = np.pi
    print(np.linalg.norm(T - y) / np.sqrt(T.shape[0]), np.max(T-y))
    #np.savetxt('ref.txt', np.hstack((x,y)))
