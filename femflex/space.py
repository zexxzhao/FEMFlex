from abc import ABC, abstractmethod
from typing import List


class GenericSpace(ABC):

    __slots__ = ('_dof', '_basis', '_mesh', '_element')

    def __init__(self, mesh, element, **kwargs):
        self._mesh = mesh
        self._element = element
        self._dof = self._impl_generate_cell_to_dof_mapping(**kwargs)
        self._basis = self._impl_generate_cell_to_basis_mapping(**kwargs)

    def mesh(self):
        return self._mesh

    def element(self):
        return self._element

    def cell_dof(self, i) -> List[int]:
        return self._dof[i]

    def cell_basis(self, i) -> List[float]:
        return self._basis[i]

    def num_dofs(self) -> int:
        return max([d for cdof in self._dof for d in cdof]) + 1

    @abstractmethod
    def _impl_generate_cell_to_dof_mapping(self, **kwargs):
        pass

    @abstractmethod
    def _impl_generate_cell_to_basis_mapping(self, **kwargs):
        pass
