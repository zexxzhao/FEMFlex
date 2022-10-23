import numpy as np
import scipy
import matplotlib.pyplot as plt

import femflex as flex       


class SpaceEnriched1DIGA(flex.GenericSpace):
    def __init__(self, mesh, porder):
        super().__init__(mesh, flex.Shape1DIGA(porder))

    def _impl_generate_cell_to_dof_mapping(self, **kwargs):
        porder = self._element.porder
        ncell = mesh.num_cell()
        dofs = [[i + j] for i in range(3) for j in range(ncell)]
        return dofs

    def _impl_generate_cell_to_basis_mapping(self, **kwargs):
        porder = self._element.porder
        ncell = mesh.num_cell()
        if porder == 2:
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

            dofs = [load_basis(ic) for ic in range(ncell)]
        else:
            raise RuntimeError("Not supported yet.\n")

rhoc = 0.5
am = 0.5 * (3 - rhoc) / (1 + rhoc)
af = 1 / (1 + rhoc)
gamma = 0.5 + am - af
dt = 1e-3
k0, k1 = 1.0, 1e2

def qr(x):
    gp = np.array([1, -1]) / np.sqrt(3)
    gw = np.array([1.0, 1.0])
    return gp, gw

def assemble(space, dT1, dT0, T0):
    ndof = space.num_dofs()
    R = np.zeros((ndof))
    ncell = space.num_cells()
    for ic in range(ncell):
        basis = space.cell_basis(ic)
        dof = space.cell_dof(ic)
        
        dT1_val = dT1[dof]
        dT0_val = dT0[dof]
        T0_val = T0[dof]
        dTm_val = am * dT1_val + (1 - am) * dT0_val
        Tm_val = T0_val + dt * af * (gamma * dT1_val + (1 - gamma) * dT0_val)

if __name__ == '__main__':
    mesh = flex.IntervalMesh(11)
    space = SpaceEnriched1DIGA(mesh, 2)
