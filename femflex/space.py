from abc import ABC, abstractmethod

class GenericSpace(ABC):
    __slots__ = ('dof', 'basis')
    def __init__(self, mesh, element, **kwargs):
        self.dof = self._impl_generate_cell_to_dof_mapping(kwargs)
        self.basis = self._impl_generate_cell_to_basis_mapping(kwargs)
        self.ncell = mesh.num_cell()
        self.element = element

    def cell_dof(self, i):
        return self.dofs[i]

    def cell_basis(self, i):
        return self.basis[i]

    @abstractmethod
    def _impl_generate_cell_to_dof_mapping(self, **kwargs):
        pass

    @abstractmethod
    def _impl_generate_cell_to_basis_mapping(self, **kwargs):
        pass


