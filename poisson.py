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


def qr(n=5):
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
        w0 = 0.5 + sqrt(30) / 36
        w1 = 0.5 - sqrt(30) / 36
        gw = asarray([w1, w0, w0, w1])
    elif n == 5:
        p0 = 1/3 * sqrt(5-2*sqrt(10/7))
        p1 = 1/3 * sqrt(5+2*sqrt(10/7))
        gp = asarray([-p1, -p0, 0.0, p0, p1])
        w0 = (322 + 13*sqrt(70))/900
        w1 = (322 - 13*sqrt(70))/900
        gw = asarray([w1, w0, 128/225, w0, w1])

    return (gp + 1) * 0.5, gw * 0.5


def temperature(x):
    pi = np.pi
    return np.sin(pi * x)


def assemble_mat(space: GenericSpace, T0: np.ndarray) -> np.ndarray:
    ndof = space.num_dofs()
    R = np.zeros((ndof, ndof))
    mesh = space.mesh()
    ncell = space.mesh().num_cells()
    element = space.element()
    gp, gw = qr()
    basis_grad_cache = element.eval(gp, order=1)
    for ic in range(ncell):
        basis_dof = space.cell_basis(ic)
        dof = space.cell_dof(ic)
        xx = mesh.cell_coordinates(ic).squeeze()
        basis_grad_val = basis_grad_cache[basis_dof]
        dxdxi = xx[1] - xx[0]
        detJ = np.abs(dxdxi)
        basis_grad_val /= dxdxi
        Rcell = basis_grad_val @ (basis_grad_val * gw * detJ).T
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


def assemble_vec(space: GenericSpace, T0: np.ndarray) -> np.ndarray:
    ndof = space.num_dofs()
    R = np.zeros((ndof,))
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
        xc = xx[0] + dxdxi * gp
        detJ = np.abs(dxdxi)
        basis_grad_val /= dxdxi
        gradTm_val = T0[dof].dot(basis_grad_val)
        Rcell = np.dot(basis_grad_val, k0 * gradTm_val * gw * detJ)\
            + np.dot(basis_val, -np.pi**2*temperature(xc) * gw * detJ)
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

        xi = np.arange(n)/n
        basis_val = element.eval(xi, order=0, index=basis_dof)
        xc = xx[0] + (xx[1] - xx[0]) * xi
        xp += xc.tolist()
        vp += v[dofs].dot(basis_val).tolist()
    return np.asarray(xp), np.asarray(vp)


def evaluate_error(v0, v1):
    n = v0.shape[0]
    return np.linalg.norm(v0 - v1) / np.sqrt(n), np.max(v0 - v1)


def main(nmesh=4, visual=True):

    mesh = flex.IntervalMesh(nmesh, 2)
    space = SpaceEnriched1DIGA(mesh, 2)
    ndof = space.num_dofs()

    T0 = np.zeros((ndof,))
    A = assemble_mat(space, T0)
    b = assemble_vec(space, T0)
    T0[:] = -np.linalg.solve(A, b)

    # print(np.linalg.det(A))
    # print(b)
    # print(T0)
    # print(assemble_vec(space, T0))
    xp, yp = evaluate_vec(space, T0)
    Tp = temperature(xp)

    if visual:
        plt.figure(figsize=(8*1.5, 3.5*1.5))
        plt.subplot(121)
        plt.plot(xp, yp, 'k', label='2nd IGA')
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
    for i in range(1, 8):
        e = main(2**i, 0)
        error_l2.append(e[0])
        print(2**i, e)
    error_l2 = np.array(error_l2)
    p = np.polyfit(np.log(1/2**np.arange(1, 8)), np.log(error_l2), 1)
    print(p)
