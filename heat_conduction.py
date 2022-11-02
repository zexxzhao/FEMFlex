import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable

import femflex as flex
from femflex.space import GenericSpace

from numba import jit


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
dt = 1e-3
k0, k1 = 1.0e0, 1.0e0


@jit(nopython=True)
def qr(n=4):
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
        gp = asarray([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
        gw = asarray([5.0, 8.0, 5.0]) / 9.0
    elif n == 4:
        p0 = sqrt(3/7-2/7*np.sqrt(6/5))
        p1 = sqrt(3/7+2/7*np.sqrt(6/5))
        gp = asarray([-p1, -p0, p0, p1])
        w0 = 0.5 - sqrt(30) / 36
        w1 = 0.5 + sqrt(30) / 36
        gw = asarray([w0, w1, w1, w0])

    return (gp + 1) * 0.5, gw * 0.5


def assemble_init(space: GenericSpace,
                  T0: np.ndarray, f: Callable) -> np.ndarray:
    ndof = space.num_dofs()
    R = np.zeros((ndof,))
    mesh = space.mesh()
    ncell = space.mesh().num_cells()
    element = space.element()
    gp, gw = qr()
    basis_val_all = element.eval(gp, order=0)
    basis_grad_val_all = element.eval(gp, order=1)
    for ic in range(ncell):
        basis_dof = space.cell_basis(ic)
        dof = space.cell_dof(ic)
        xx = mesh.cell_coordinates(ic).squeeze()
        basis_val = basis_val_all[basis_dof]
        basis_grad_val = basis_grad_val_all[basis_dof]
        dxdxi = xx[1] - xx[0]
        xc = xx[0] + dxdxi * gp
        dxdxi = np.dot(xx, basis_grad_val)
        detJ = np.abs(dxdxi)
        basis_grad_val /= dxdxi
        T0_val = T0[dof].dot(basis_val)

        Rcell = np.dot(basis_val, (T0_val - f(0, xc)) * gw * detJ)
        R[dof] += Rcell
    return R[1:-1]


def assemble_mat(space: GenericSpace) -> np.ndarray:
    ndof = space.num_dofs()
    R = np.zeros((ndof, ndof))
    mesh = space.mesh()
    ncell = space.mesh().num_cells()
    element = space.element()
    gp, gw = qr()
    basis_cache = element.eval(gp, order=0)
    basis_grad_cache = element.eval(gp, order=1)
    for ic in range(ncell):
        basis_dof = space.cell_basis(ic)
        dof = space.cell_dof(ic)
        xx = mesh.cell_coordinates(ic).squeeze()
        basis_val = basis_cache[basis_dof]
        basis_grad_val = basis_grad_cache[basis_dof]
        dxdxi = xx[1] - xx[0]
        detJ = np.abs(dxdxi)
        basis_grad_val /= dxdxi
        Rcell = am * basis_val @ (basis_val * gw * detJ).T
        Rcell += dt * af * gamma \
            * basis_grad_val @ (basis_grad_val * gw * detJ).T
        R[np.ix_(dof, dof)] += Rcell
    # print(R)
    # sys.exit()
    R[0, :] = 0.0
    R[:, 0] = 0.0
    R[0, 0] = 1.0
    R[:, -1] = 0.0
    R[-1, :] = 0.0
    R[-1, -1] = 1.0
    return R


def assemble_vec(space: GenericSpace, dT1: np.ndarray,
                 dT0: np.ndarray, T0: np.ndarray, src) -> np.ndarray:
    ndof = space.num_dofs()
    R = np.zeros((ndof,))
    mesh = space.mesh()
    ncell = space.mesh().num_cells()
    element = space.element()
    gp, gw = qr()
    basis_cache = element.eval(gp, order=0)
    basis_grad_cache = element.eval(gp, order=1)
    dTm = am * dT1 + (1 - am) * dT0
    Tm = T0 + dt * af * (gamma * dT1 + (1 - gamma) * dT0)

    for ic in range(ncell):
        basis_dof = space.cell_basis(ic)
        dof = space.cell_dof(ic)
        xx = mesh.cell_coordinates(ic).squeeze()
        basis_val = basis_cache[basis_dof]
        basis_grad_val = basis_grad_cache[basis_dof]
        dxdxi = xx[1] - xx[0]
        xc = xx[0] + dxdxi * gp
        detJ = np.abs(dxdxi)
        basis_grad_val /= dxdxi
        dTm_val = dTm[dof].dot(basis_val)
        gradTm_val = Tm[dof].dot(basis_grad_val)
        Rcell = np.dot(basis_val, dTm_val * gw * detJ)
        Rcell += np.dot(basis_grad_val, k0 * gradTm_val * gw * detJ)\
            - np.dot(basis_val, src(xc) * gw * detJ)
        R[dof] += Rcell
    # print(R)
    # sys.exit()
    R[0] = 0.0
    R[-1] = 0.0
    return R


def evaluate_vec(space, v, n=100):
    mesh = space.mesh()
    xp = []
    vp = []
    element = space.element()
    for ic in range(mesh.num_cells()):
        xx = mesh.cell_coordinates(ic).squeeze()
        basis_dof = space.cell_basis(ic)
        dofs = space.cell_dof(ic)

        xi = np.linspace(0, 1, n+1)
        basis_val = element.eval(xi, order=0, index=basis_dof)
        xc = xx[0] + (xx[1] - xx[0]) * xi
        xp += xc.tolist()
        vp += v[dofs].dot(basis_val).tolist()
    return np.asarray(xp), np.asarray(vp)


def evaluate_error(v0, v1):
    n = v0.shape[0]
    return np.linalg.norm(v0 - v1) / np.sqrt(n), np.max(np.abs(v0 - v1))


def src(x):
    pi = np.pi
    return np.sin(pi*x)


def temperature(t, x):
    pi = np.pi
    return (1 - np.exp(-pi**2*t))/pi**2 * np.sin(pi*x)


def main(nmesh, nntime=1000, visual=True):
    mesh = flex.IntervalMesh(nmesh, 2)
    space = SpaceEnriched1DIGA(mesh, 2)
    ndof = space.num_dofs()
    dT1 = np.zeros((ndof,))
    dT0 = np.zeros((ndof,))
    T0 = np.zeros((ndof,))

    A = assemble_mat(space)
    for i in range(nntime):
        # print(f'i={i+1}, time={i*dt+dt}')
        dT1 *= (gamma - 1) / gamma
        b = assemble_vec(space, dT1, dT0, T0, src)
        dT1[:] -= np.linalg.solve(A, b)
        # res = assemble_vec(space, dT1, dT0, T0, src)
        # print(np.linalg.norm(res))
        T0[:] += dt * (gamma * dT1 + (1 - gamma) * dT0)
        dT0[:] = dT1[:]

    xp, yp = evaluate_vec(space, T0)
    Tp = temperature(1.0, xp)

    if visual:
        plt.figure(figsize=(8*1.5, 3.5*1.5))
        plt.subplot(121)
        plt.plot(xp, yp, 'k', label='IGA')
        plt.plot(xp, Tp, 'r', label='Exact')
        plt.xlabel('x', fontsize=20)
        plt.ylabel(r'${T}$', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim([0, 1])
        plt.legend(fontsize=15)
        plt.tight_layout()
        plt.subplot(122)
        plt.plot(xp, Tp - yp)
        plt.xlabel('x', fontsize=20)
        plt.ylabel(r'${T}$', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim([0, 1])
        plt.tight_layout()
        plt.savefig('./cmp.png')

    return evaluate_error(Tp, yp)


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
    error_l2 = []
    for i in range(1, 7):
        ncell = 2 ** i
        e = main(ncell, 1000, 0)
        error_l2.append(e[0])
        print(i, e)
    error_l2 = np.array(error_l2)
    p = np.polyfit(np.log(1/2**np.arange(1, 7)), np.log(error_l2), 1)
    print(p)
