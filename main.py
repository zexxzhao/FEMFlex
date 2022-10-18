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
        dofs = [[] for _ in range(ncell)]
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
        
if __name__ == '__main__':
    mesh = flex.IntervalMesh(11)
    space = SpaceEnriched1DIGA(mesh, 2)
