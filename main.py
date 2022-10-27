import numpy as np
import matplotlib.pyplot as plt

import femflex as flex


class SpaceEnriched1DIGA(flex.GenericSpace):
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
dt = 1e-2
k0, k1 = 1.0e2, 1.0e2


def qr(x):
    gp = np.array([1, -1]) / np.sqrt(3)
    gp = (gp + 1) * 0.5
    gw = np.array([1.0, 1.0]) * 0.5
    return gp, gw


def assemble(space, dT1, dT0, T0):
    ndof = space.num_dofs()
    R = np.zeros((ndof))
    mesh = space.mesh()
    ncell = space.mesh().num_cells()
    element = space._element
    for ic in range(ncell):
        basis_dof = space.cell_basis(ic)
        dof = space.cell_dof(ic)
        xx = mesh.node(mesh.cell(ic))
        gp, gw = qr(xx)
        detJ = abs(xx[1] - xx[0])
        basis_val = np.array(
            [[element.eval(p, order=0, index=fid) for p in gp]
                for fid in basis_dof])
        basis_grad_val = np.array(
            [[element.eval(p, order=1, index=fid) for p in gp]
                for fid in basis_dof])
        dT1_val = np.dot(dT1[dof], basis_val)
        dT0_val = np.dot(dT0[dof], basis_val)
        dTm_val = am * dT1_val + (1 - am) * dT0_val
        gradTm_val = T0[dof].dot(basis_grad_val)
        + dt * af * (gamma * dT1[dof].dot(basis_grad_val)
                     + (1 - gamma) * dT0[dof].dot(basis_grad_val))
        Rcell = np.dot(basis_val, dTm_val * gw) * detJ \
            + np.dot(basis_grad_val, k0 * gradTm_val * gw) * detJ
        R[dof] += Rcell
    # print(np.linalg.norm(R))
    return R


def evaluate_vec(space, v, xp):
    mesh = space.mesh()
    x = mesh.node()
    vp = np.zeros_like(xp)
    for ix, point in enumerate(xp):
        ic = np.argwhere(
            np.squeeze((x <= point) * (point <= np.roll(x, -1))))[0]
        ic = np.squeeze(ic)
        xcell = mesh.cell_coordinates(ic)
        xi = (point - xcell[0]) / (xcell[1] - xcell[0])
        basis_dof = space.cell_basis(ic)
        basis_val = space.element().eval(xi, 0, basis_dof)
        dof = space.cell_dof(ic)
        vp[ix] = v[dof].dot(basis_val)
    return vp


if __name__ == '__main__':
    mesh = flex.IntervalMesh(11)
    space = SpaceEnriched1DIGA(mesh, 2)
    ndof = space.num_dofs()
    dT1 = np.zeros((ndof,))
    dT0 = np.zeros((ndof,))
    T0 = np.zeros((ndof,))

    for i in range(ndof):
        T0[i] = np.sin(np.pi*i/(ndof-1)) * 1.0
    T0[-1] = 0.0
    nntime = 10
    from scipy.optimize import root
    for i in range(nntime):
        print(f'i={i}, time={i*dt}')
        dT1 *= (gamma - 1) / gamma
        sol = root(lambda dTem: assemble(space, dTem, dT0, T0),
                   dT1, tol=1e-6)
        dT1[:] = sol.x
        T0[:] += dt * (gamma * dT1 + (1 - gamma) * dT0)
        np.savetxt(f'dat/tem{i}.txt', T0)
    visual = True
    if visual:
        x = np.linspace(0, 1, 101)
        Tp = evaluate_vec(space, T0, x)
        plt.plot(x, Tp)
        plt.show()
