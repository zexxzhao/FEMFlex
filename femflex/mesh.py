from abc import ABC, abstractmethod


class MeshBase(ABC):
    __slots__ = ['nodes', 'cells', 'facets']

    def __init__(self, **kwargs):
        self.nodes = self._impl_generate_nodes(**kwargs)
        self.cells = self._impl_generate_cells(**kwargs)
        self.facets = self._impl_generate_facets(**kwargs)

    def cell(self, index=None):
        if index is None:
            return self.cells
        return self.cells[index]

    def node(self, index=None):
        if index is None:
            return self.nodes
        return self.nodes[index]

    def cell_coordinates(self, i):
        return self.nodes[self.cells[i]]

    def facet(self, i):
        return self.facets[i]

    def num_cells(self):
        return len(self.cells)

    def num_nodes(self):
        return len(self.nodes)

    def num_facets(self):
        return len(self.facets)

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
    def __init__(self, n, degree):
        super().__init__(n=n, degree=degree)

    def _impl_generate_nodes(self, **kwargs):
        n = kwargs['n'] if 'n' in kwargs else 0
        deg = kwargs['degree'] if 'degree' in kwargs else 2
        from numpy import linspace
        return linspace(0, 1, n + deg).reshape(-1, 1)

    def _impl_generate_cells(self, **kwargs):
        n = kwargs['n'] if 'n' in kwargs else 0
        deg = kwargs['degree'] if 'degree' in kwargs else 2
        return [[i + j for j in range(deg + 1)] for i in range(n)]

    def _impl_generate_facets(self, **kwargs):
        return []
