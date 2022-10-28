import numpy as np
import matplotlib.pyplot as plt

import femflex as flex
from femflex.space import GenericSpace


class SpaceEnriched1DIGA(GenericSpace):
    def __init__(self, mesh, k):
        super().__init__(mesh, flex.Shape1DIGA(k))

    def _impl_generate_cell_to_dof_mapping(self, **kwargs):
        ncell = self.mesh().num_cells()
        dofs = [[i + j for j in range(3)] for i in range(ncell)]
        return dofs

    def _impl_generate_cell_to_basis_mapping(self, **kwargs):
        k = self.element().degree
        ncell = self.mesh().num_cells()
        if k == 2:
            def load_basis(n):
                if n == 0:
                    return [0, 1, 3]
                elif n == 2:
                    return [2, 4, 3]
                elif n == ncell - 2:
                    return [5, 4, 6]
                elif n == ncell - 1:
                    return [5, 7, 8]
                else:
                    return [5, 4, 3]

            basis_fn_id = [load_basis(ic) for ic in range(ncell)]
        else:
            raise RuntimeError("Not supported yet.\n")
        return basis_fn_id


rhoc = 0.5
am = 0.5 * (3 - rhoc) / (1 + rhoc)
af = 1 / (1 + rhoc)
gamma = 0.5 + am - af
dt = 1e-3/16
k0, k1 = 1.0e0, 1.0e0


def qr(n=3):
    sqrt = np.sqrt
    asarray = np.asarray
    if n == 1:
        gp = asarray([0.0])
        gw = asarray([2.0])
    elif n == 2:
        p = sqrt(1/3)
        gp = asarray([p, -p])
        gw = asarray([1.0, 1.0])
    elif n == 3:
        gp = asarray([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
        gw = asarray([5, 8, 5]) / 9
    elif n == 4:
        p0 = sqrt(3/7-2/7*np.sqrt(6/5))
        p1 = sqrt(3/7+2/7*np.sqrt(6/5))
        gp = asarray([-p1, -p0, p0, p1])
        w0 = 0.5 - np.sqrt(30) / 36
        w1 = 0.5 + np.sqrt(30) / 36
        gw = asarray([w0, w1, w1, w0])

    return (gp + 1) * 0.5, gw * 0.5


def assemble(space: GenericSpace, dT1: np.ndarray,
             dT0: np.ndarray, T0: np.ndarray) -> np.ndarray:
    ndof = space.num_dofs()
    R = np.zeros((ndof,))
    mesh = space.mesh()
    ncell = space.mesh().num_cells()
    element = space.element()
    for ic in range(ncell):
        basis_dof = space.cell_basis(ic)
        dof = space.cell_dof(ic)
        xx = mesh.cell_coordinates(ic)
        gp, gw = qr()
        basis_val = element.eval(gp, order=0, index=basis_dof)
        basis_grad_val = element.eval(gp, order=1, index=basis_dof)

        dxdxi = np.dot(xx.squeeze(), basis_grad_val)
        detJ = np.abs(dxdxi)
        basis_grad_val /= dxdxi
        dT1_val = np.dot(dT1[dof], basis_val)
        dT0_val = np.dot(dT0[dof], basis_val)
        dTm_val = am * dT1_val + (1 - am) * dT0_val
        gradTm_val = T0[dof].dot(basis_grad_val)
        + dt * af * (gamma * dT1[dof].dot(basis_grad_val)
                     + (1 - gamma) * dT0[dof].dot(basis_grad_val))
        Rcell = np.dot(basis_val, dTm_val * gw * detJ) \
            + np.dot(basis_grad_val, k0 * gradTm_val * gw * detJ)
        R[dof] += Rcell
    return R


def evaluate_vec(space, v, n=100):
    mesh = space.mesh()
    xp = []
    vp = []
    element = space.element()
    for ic in range(mesh.num_cells()):
        xc = mesh.cell_coordinates(ic).squeeze()
        basis_dof = space.cell_basis(ic)
        dofs = space.cell_dof(ic)

        xi = np.linspace(0, 1, n+1)
        basis_val = element.eval(xi, order=0, index=basis_dof)
        xp += xc.dot(basis_val).tolist()
        vp += v[dofs].dot(basis_val).tolist()
    return xp, vp


def evaluate_error(v0, v1):
    n = v0.shape[0]
    return np.linalg.norm(v0 - v1) / np.sqrt(n)


def main(nntime=100, visual=True):
    mesh = flex.IntervalMesh(40, 2)
    space = SpaceEnriched1DIGA(mesh, 2)
    ndof = space.num_dofs()
    dT1 = np.zeros((ndof,))
    dT0 = np.zeros((ndof,))
    T0 = np.zeros((ndof,))
    for i in range(ndof):
        T0[i] = np.sin(np.pi*i/(ndof-1)) * 1.0
    T0[-1] = 0.0
    Tinit = T0.copy()
    from scipy.optimize import root
    for i in range(nntime):
        print(f'i={i+1}, time={i*dt+dt}')
        dT1 *= (gamma - 1) / gamma
        sol = root(lambda dTem: assemble(space, dTem, dT0, T0),
                   dT1, tol=1e-6, method='hybr')

        dT1[:] = sol.x
        T0[:] += dt * (gamma * dT1 + (1 - gamma) * dT0)
        dT0[:] = dT1[:]
    np.savetxt(f'tem{i+1}.txt', T0)

    if visual:
        plt.figure(figsize=(8*1.5, 3.5*1.5))
        plt.subplot(121)
        xp, yp = evaluate_vec(space, dT1)
        plt.plot(xp, yp, 'k')
        plt.xlabel('x', fontsize=20)
        plt.ylabel(r'$\dot{T}$', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim([0, 1])
        plt.tight_layout()
        plt.subplot(122)
        xp, yp = evaluate_vec(space, Tinit)
        plt.plot(xp, yp, 'b', label='t=0.00')
        xp, yp = evaluate_vec(space, T0)
        plt.plot(xp, yp, 'k', label='t=0.05')
        ref = np.loadtxt('ref.txt')
        plt.plot(ref[:, 0], ref[:, 1], 'r', label='FENICS')
        plt.xlabel('x', fontsize=20)
        plt.ylabel(r'${T}$', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim([0, 1])
        plt.legend(fontsize=15)
        plt.tight_layout()
        plt.savefig('./cmp.png')
        Tp = np.interp(xp, ref[:, 0], ref[:, 1])
        print(evaluate_error(Tp, yp))


def profiling(f, turn_on):
    def func(*args):
        import yappi
        yappi.set_clock_type("cpu")
        yappi.start()
        ret = f(*args)
        yappi.stop()
        yappi.get_func_stats().print_all(out=open('prof.txt', 'w'))
        return ret
    return func if turn_on else f


if __name__ == '__main__':
    main = profiling(main, 0)
    main(800)
