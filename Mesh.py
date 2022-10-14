from abc import ABC, abstractmethod
class MeshBase(ABC):
    __slots__ = ['nodes', 'cells', 'facets']

    def __init__(self, **kwargs):
        self.nodes = self._impl_generate_nodes(kwargs)
        self.cells = self._impl_genereate_cells(kwargs)
        self.facets = self._impl_generate_facets(kwargs)

    def cell(self, i):
        return self.cells[i]

    def node(self, i):
        return self.nodes[i]

    def facet(self, i):
        return self.facets[i]

    @abstractmethod
    def _impl_generate_nodes(self, **kwargs):
        pass

    @abstractmethod
    def _impl_generate_cells(self, **kwargs):
        pass

    @abstractmethod
    def _impl_generate_facets(self, **kwargs):
        pass

class IntervalMesh(MeshBase):
    def __init__(self, n):
        super().__init__(n);

    def _impl_generate_nodes(self, n):
        from numpy import linspace
        return linspace(0, 1, n + 1).reshape(-1, 1)

    def _impl_generate_cells(self, n):
        self.element = [[i, i+1] for i in range(n)] 

    def _impl_generate_facets(self, n):
        self.facets = [[i] for range(n + 1)]

