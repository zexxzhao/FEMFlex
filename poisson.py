import sys
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


def assemble(space: GenericSpace, T0: np.ndarray) -> np.ndarray:
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
        xc = xx.dot(basis_val)
        dxdxi = np.dot(xx.squeeze(), basis_grad_val)
        detJ = np.abs(dxdxi)
        basis_grad_val /= dxdxi
        gradTm_val = T0[dof].dot(basis_grad_val)
        Rcell = np.dot(basis_grad_val, k0 * gradTm_val * gw * detJ)\
            + np.dot(basis_val, -np.pi**2*temperature(xc) * gw * detJ)
        R[dof] += Rcell
        print(Rcell)
    print(R)
    sys.exit()
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
    return np.asarray(xp), np.asarray(vp)


def evaluate_error(v0, v1):
    n = v0.shape[0]
    return np.linalg.norm(v0 - v1) / np.sqrt(n), np.max(v0 - v1)


def main(nntime=100, visual=True):
    nmesh = 2
    import sys
    if len(sys.argv) == 2:
        nmesh = int(sys.argv[1])
    mesh = flex.IntervalMesh(nmesh, 2)
    space = SpaceEnriched1DIGA(mesh, 2)
    ndof = space.num_dofs()

    T0 = np.zeros((ndof,))

    from scipy.optimize import root
    sol = root(lambda x: assemble(space, x), T0, tol=1e-15)
    T0[:] = sol.x
    T0 -= np.mean(T0)
    if visual:
        plt.figure(figsize=(8*1.5, 3.5*1.5))
        plt.subplot(121)
        xp, yp = evaluate_vec(space, T0)
        plt.plot(xp, yp, 'k', label='2nd IGA')
        Tp = temperature(xp)
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
        print(*evaluate_error(Tp, yp))


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
    main(5*16)
