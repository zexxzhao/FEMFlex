from abc import ABC, abstractmethod

class GenericSpace(ABC):
    __slots__ = ('_dof', '_basis', '_ncell', '_element')
    def __init__(self, mesh, element, **kwargs):
        self._ncell = mesh.num_cell()
        self._mesh = mesh
        self._element = element
        self._dof = self._impl_generate_cell_to_dof_mapping(**kwargs)
        self._basis = self._impl_generate_cell_to_basis_mapping(**kwargs)

    def cell_dof(self, i):
        return self._dof[i]

    def cell_basis(self, i):
        return self._basis[i]
    
    def num_dofs(self):
        ncell = self._mesh.num_cell()
        return max([d for ic in range(ncell) for d in self.cell_dof(ic)])

    @abstractmethod
    def _impl_generate_cell_to_dof_mapping(self, **kwargs):
        pass

    @abstractmethod
    def _impl_generate_cell_to_basis_mapping(self, **kwargs):
        pass

