import sys # NOQA
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
from typing import List

import femflex as flex
from femflex.space import GenericSpace


class SpaceEnriched1DIGA(GenericSpace):
    def __init__(self, mesh, k, nval):
        super().__init__(mesh, flex.Shape1DIGA(k), nval=nval)

    def _impl_generate_cell_to_dof_mapping(self, **kwargs):
        ncell = self.mesh().num_cells()
        deg = self.element().degree
        nval = 1
        if 'nval' in kwargs:
            nval = kwargs['nval']
        dofs = [[(i + j) * nval + k
                 for j in range(1 + deg)
                 for k in range(nval)]
                for i in range(ncell)]
        return dofs

    def _impl_generate_cell_to_basis_mapping(self, **kwargs):
        k = self.element().degree
        ncell = self.mesh().num_cells()
        if k == 2:
            def load_basis(n):
                if n == 0:
                    return [0, 1, 3]
                elif n == ncell - 1:
                    return [5, 7, 8]
                elif n == ncell - 2:
                    return [5, 4, 6]
                elif n == 2:
                    return [2, 4, 3]
                else:
                    return [5, 4, 3]

            basis_fn_id = [load_basis(ic) for ic in range(ncell)]
        else:
            raise RuntimeError("Not supported yet.\n")
        return basis_fn_id


class Dirichlet(object):
    def __init__(self, dofs):
        self._dofs = dofs

    @property
    def dofs(self):
        return self._dofs

    def __call__(self, **kwargs):
        dof = self.dofs
        if 'A' in kwargs:
            A = kwargs['A']
            A[dof, :] = 0.0
            A[:, dof] = 0.0
            A[dof, dof] = 1.0
            return A
        if 'x' in kwargs:
            x = kwargs['x']
            value = x[dof] * 0.0
            if 'value' in kwargs and kwargs['value'] is not None:
                value = kwargs['value']
            x[dof] = value
            return x
        return


rhoc = 0.5
am = 0.5 * (3 - rhoc) / (1 + rhoc)
af = 1 / (1 + rhoc)
gamma = 0.5 + am - af
dt = 1e-3
k0, k1 = 1.0e0, 1.0e0
tauB = 1e3


def assemble_init(space: GenericSpace,
                  T0: np.ndarray, f: Callable) -> np.ndarray:
    ndof = space.num_dofs()
    R = np.zeros((ndof,))
    mesh = space.mesh()
    ncell = space.mesh().num_cells()
    element = space.element()
    gp, gw = flex.qr()
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


class Assembler:
    def __init__(self, space: GenericSpace, bcs: List[Dirichlet] = None):
        self.space = space
        self.bcs = [] if bcs is None else bcs

    @staticmethod
    def sum(x, y):
        return np.dot(x, y.T)

    def assemble_mat(self) -> np.ndarray:
        space = self.space
        ndof = space.num_dofs()
        R = np.zeros((ndof, ndof))
        mesh = space.mesh()
        ncell = space.mesh().num_cells()
        element = space.element()
        qr_p, qr_w = flex.qr(4)
        basis_cache = element.eval(qr_p, order=0)
        basis_grad_cache = element.eval(qr_p, order=1)

        def is_cut_element(x):
            dx = xx[1] - xx[0]
            return xx[0] + dx*1e-3 < 0.5 < xx[1] - dx*1e-3

        for ic in range(ncell):
            basis_dof = space.cell_basis(ic)
            basis_dof2 = np.repeat(basis_dof, 2)
            dof = space.cell_dof(ic)
            xx = mesh.cell_coordinates(ic).squeeze()
            dxdxi = xx[1] - xx[0]
            xc = xx[0] + dxdxi * qr_p
            basis_val = basis_cache[basis_dof2]
            basis_grad_val = basis_grad_cache[basis_dof2]
            weights = qr_w[:]
            if is_cut_element(xx):
                gp = np.hstack((qr_p * 0.5, qr_p * 0.5 + 0.5))
                xc = xx[0] + (xx[1] - xx[0]) * gp
                basis_val = element.eval(gp, order=0, index=basis_dof)
                basis_val = np.repeat(basis_val, 2, axis=0)
                basis_grad_val = element.eval(gp, order=1, index=basis_dof)
                basis_grad_val = np.repeat(basis_grad_val, 2, axis=0)
                weights = np.hstack((weights, weights)) * 0.5

            basis_val[0::2] *= xc < 0.5
            basis_val[1::2] *= xc >= 0.5
            basis_grad_val[0::2] *= xc < 0.5
            basis_grad_val[1::2] *= xc >= 0.5

            detJ = np.abs(dxdxi)
            basis_grad_val /= dxdxi
            Rcell = am * basis_val @ (basis_val * weights * detJ).T
            K = np.diag(np.tile([k0, k1], 3))
            Rcell += dt * af * gamma \
                * K @ basis_grad_val @ (basis_grad_val * weights * detJ).T

            # enrichment here
            if xx[0] + dxdxi*1e-3 < 0.5 < xx[1] - dxdxi*1e-3:
                gp = np.asarray([0.5])
                basis_val = element.eval(gp, order=0, index=basis_dof)
                basis_grad_val = element.eval(gp, order=1, index=basis_dof)
                c = dt * af * gamma
                Rcell[0::2, 0::2] -= c * self.sum(basis_val,
                                                  0.5 * k0 * basis_grad_val)
                Rcell[0::2, 0::2] -= c * self.sum(k0 * basis_grad_val,
                                                  0.5 * basis_val)
                Rcell[0::2, 0::2] += c * self.sum(tauB * basis_val, basis_val)

                Rcell[0::2, 1::2] -= c * self.sum(basis_val,
                                                  0.5 * k1 * basis_grad_val)
                Rcell[0::2, 1::2] += c * self.sum(k0 * basis_grad_val,
                                                  0.5 * basis_val)
                Rcell[0::2, 1::2] -= c * self.sum(tauB * basis_val, basis_val)

                Rcell[1::2, 0::2] += c * self.sum(basis_val,
                                                  0.5 * k0 * basis_grad_val)
                Rcell[1::2, 0::2] -= c * self.sum(k1 * basis_grad_val,
                                                  0.5 * basis_val)
                Rcell[1::2, 0::2] -= c * self.sum(tauB * basis_val, basis_val)

                Rcell[1::2, 1::2] += c * self.sum(basis_val,
                                                  0.5 * k1 * basis_grad_val)
                Rcell[1::2, 1::2] += c * self.sum(k1 * basis_grad_val,
                                                  0.5 * basis_val)
                Rcell[1::2, 1::2] += c * self.sum(tauB * basis_val,
                                                  basis_val)

            R[np.ix_(dof.T.flatten(), dof.T.flatten())] += Rcell
        return R

    def assemble_vec(self, dT1: np.ndarray,
                     dT0: np.ndarray, T0: np.ndarray,
                     src: Callable) -> np.ndarray:
        space = self.space
        ndof = space.num_dofs()
        R = np.zeros((ndof,))
        mesh = space.mesh()
        ncell = space.mesh().num_cells()
        element = space.element()
        qr_p, qr_w = flex.qr(4)
        basis_cache = element.eval(qr_p, order=0)
        basis_grad_cache = element.eval(qr_p, order=1)

        def is_cut_element(x):
            dx = xx[1] - xx[0]
            return xx[0] + dx*1e-3 < 0.5 < xx[1] - dx*1e-3

        dTm = am * dT1 + (1 - am) * dT0
        Tm = T0 + dt * af * (gamma * dT1 + (1 - gamma) * dT0)
        for ic in range(ncell):
            basis_dof = space.cell_basis(ic)
            # basis_dof2 = np.repeat(basis_dof, 2)
            dof = space.cell_dof(ic)
            xx = mesh.cell_coordinates(ic).squeeze()
            basis_val = basis_cache[basis_dof]
            basis_grad_val = basis_grad_cache[basis_dof]
            dxdxi = xx[1] - xx[0]
            xc = xx[0] + dxdxi * qr_p
            weights = qr_w[:]
            if is_cut_element(xx):
                gp = np.hstack((qr_p * 0.5, qr_p * 0.5 + 0.5))
                xc = xx[0] + dxdxi * gp
                basis_val = element.eval(gp, order=0, index=basis_dof)
                basis_grad_val = element.eval(gp, order=1, index=basis_dof)
                weights = np.tile(qr_w, 2) * 0.5
            weights = np.tile(weights, 2).reshape(2, -1)
            weights[0, :] *= xc < 0.5
            weights[1, :] *= xc >= 0.5
            detJ = np.abs(dxdxi)
            basis_grad_val /= dxdxi
            dTm_val = dTm[dof].dot(basis_val)
            gradTm_val = Tm[dof].dot(basis_grad_val)
            Rcell = np.zeros_like(dof.T, dtype=np.float64)
            Rcell += self.sum(basis_val, dTm_val * weights * detJ)
            K = np.diag([k0, k1])
            Rcell += self.sum(basis_grad_val,
                              K @ gradTm_val * weights * detJ)
            Rcell -= self.sum(basis_val, src(xc) * weights * detJ)

            if xx[0] + dxdxi*1e-3 < 0.5 < xx[1] - dxdxi*1e-3 and 0:
                gp = np.asarray([0.5])
                basis_val = element.eval(gp, order=0, index=basis_dof)
                basis_grad_val = element.eval(gp, order=1, index=basis_dof)
                weights = np.asarray([1.0])
                flux_m = 0.5 * (k0 * Tm[dof[0, :]].dot(basis_grad_val)
                                + k1 * Tm[dof[1, :]].dot(basis_grad_val))
                tem_m = 0.5 * (Tm[dof[0, :]].dot(basis_val)
                               - Tm[dof[1, :]].dot(basis_val))
                Rcell[:, 0] -= self.sum(basis_val, flux_m)
                Rcell[:, 0] -= self.sum(k0 * basis_grad_val, tem_m)
                Rcell[:, 0] += tauB * self.sum(basis_val, tem_m)

                Rcell[:, 1] += self.sum(basis_val, flux_m)
                Rcell[:, 1] -= self.sum(k1 * basis_grad_val, tem_m)
                Rcell[:, 1] -= tauB * self.sum(basis_val, tem_m)
            R[dof.T.flatten()] += Rcell.flatten()
        return R

    def apply_bcs(self, A=None, x=None, values=None):
        if A is not None:
            for bc in self.bcs:
                bc(A=A)
        elif x is not None:
            for bc in self.bcs:
                bc(x=x, values=values)


def evaluate_vec(space, v, n=100):
    mesh = space.mesh()
    xp = []
    vp = []
    element = space.element()
    xi = np.linspace(0, 1, n+1)
    basis_cache = element.eval(xi, order=0)
    for ic in range(mesh.num_cells()):
        xx = mesh.cell_coordinates(ic).squeeze()
        basis_dof = space.cell_basis(ic)
        dofs = space.cell_dof(ic)
        basis_val = basis_cache[basis_dof]
        xc = xx[0] + (xx[1] - xx[0]) * xi
        xp += xc.tolist()
        tmp = v[dofs].dot(basis_val)
        tmp[0, :] *= xc < 0.5
        tmp[1, :] *= xc >= 0.5
        tmp = np.sum(tmp, axis=0).tolist()
        vp += tmp
    return np.asarray(xp), np.asarray(vp)


def evaluate_error(v0, v1):
    n = v0.shape[0]
    return np.linalg.norm(v0 - v1) / np.sqrt(n), np.max(np.abs(v0 - v1))


def src(x):
    pi = np.pi
    return np.ones((2, 1)) @ (np.sin(pi*x).reshape(1, -1))


def temperature(t, x):
    pi = np.pi
    return (1 - np.exp(-pi**2*t))/pi**2 * np.sin(pi*x)


def main(nmesh, nntime=1000, visual=True):
    mesh = flex.IntervalMesh(nmesh, 2)
    space = SpaceEnriched1DIGA(mesh, 2, 2)
    ndof = space.num_dofs()
    dT1 = np.zeros((ndof,))
    dT0 = np.zeros((ndof,))
    T0 = np.zeros((ndof,))
    # bc
    frozen_dofs = []
    for ic in range(nmesh):
        dof = space.cell_dof(ic)
        xx = mesh.cell_coordinates(ic).squeeze()
        dxdxi = xx[1] - xx[0]
        if xx[1] + dxdxi * 1e-3 < 0.5:
            frozen_dofs += dof[0, :].tolist()
        elif xx[0] - dxdxi * 1e-3 > 0.5:
            frozen_dofs += dof[1, :].tolist()
        elif xx[0] + dxdxi * 1e-3 < 0.5 < xx[1] - dxdxi * 1e-3:
            frozen_dofs += dof.flatten().tolist()
    alldof = np.zeros((ndof,))
    alldof[frozen_dofs] = 1
    frozen_dofs = np.argwhere(alldof == 0).squeeze().tolist()
    frozen_dofs.append(0)
    frozen_dofs.append(ndof - 1)
    active_dofs = [d for d in range(ndof) if d not in frozen_dofs]
    bc_redundant_dofs = Dirichlet(frozen_dofs)

    assembler = Assembler(space, [bc_redundant_dofs])

    A = assembler.assemble_mat()
    assembler.apply_bcs(A=A)
    from scipy.sparse import csr_matrix
    A = csr_matrix(A)
    # print(np.linalg.det(A))
    # print(A)
    print(frozen_dofs)

    # sys.exit()
    from scipy.sparse.linalg import gmres
    from scipy.optimize import root
    solver_type = 'krylov'
    for i in range(nntime):

        dT1 *= (gamma - 1) / gamma
        if solver_type == 'hybr':
            def f(x):
                tmp = np.zeros((ndof,))
                tmp[active_dofs] = x
                b = assembler.assemble_vec(dT1=tmp, dT0=dT0, T0=T0, src=src)
                return b[active_dofs]
            sol = root(f, dT1[active_dofs], tol=1e-12)
            dT1[active_dofs] = sol.x
        elif solver_type == 'krylov':
            b = assembler.assemble_vec(dT1=dT1, dT0=dT0, T0=T0, src=src)
            assembler.apply_bcs(x=b)
            dT1[:] -= gmres(A, b, tol=1e-15)[0]
        if (i+1) % 100 == 0:
            print(f'i={i+1}, time={i*dt+dt}')
            res = assembler.assemble_vec(dT1=dT1, dT0=dT0, T0=T0, src=src)
            assembler.apply_bcs(x=res)
            print(np.linalg.norm(res))
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
        # plt.savefig('./cmp.png')
        plt.show()

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
    main = profiling(main, 1)
    error_l2 = []
    rsl_range = np.arange(7, 8)
    for i in rsl_range:
        ncell = 2 ** i + 1
        e = main(ncell, 1000, 1)
        error_l2.append(e[0])
        print(i, e)
    error_l2 = np.array(error_l2)
    h = 1 / 2 ** rsl_range
    p = np.polyfit(np.log(h)[2:], np.log(error_l2)[2:], 1)
    print(p)

    plt.loglog(h, error_l2, '-*', label='2nd IGA')
    plt.loglog(h, 0.02*h**3, '--k', label='k=3')
    plt.loglog(h, 0.02*h**2, '--k', label='k=2')
    plt.xlabel(r'$h$', fontsize=20)
    plt.ylabel(r'$e$', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
