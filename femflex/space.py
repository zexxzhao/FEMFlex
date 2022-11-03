from abc import ABC, abstractmethod


class GenericSpace(ABC):

    __slots__ = ('_ndof', '_dof', '_basis', '_mesh', '_element', '_nval')

    def __init__(self, mesh, element, **kwargs):
        self._mesh = mesh
        self._element = element
        self._nval = kwargs['nval']
        self._dof = self._impl_generate_cell_to_dof_mapping(**kwargs)
        self._ndof = max([d for cdof in self._dof for d in cdof]) + 1
        self._basis = self._impl_generate_cell_to_basis_mapping(**kwargs)

    def mesh(self):
        return self._mesh

    def element(self):
        return self._element

    def nval(self):
        return self._nval

    def cell_dof(self, i):
        from numpy import asarray
        dof = asarray(self._dof[i])
        nval = self.nval()
        return dof.reshape(-1, nval).T.squeeze()

    def cell_basis(self, i):
        from numpy import asarray
        return asarray(self._basis[i])

    def num_dofs(self) -> int:
        return self._ndof

    @abstractmethod
    def _impl_generate_cell_to_dof_mapping(self, **kwargs):
        pass

    @abstractmethod
    def _impl_generate_cell_to_basis_mapping(self, **kwargs):
        pass
