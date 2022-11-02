from abc import ABC, abstractmethod
from typing import Optional, List


IntegerType = int
ExIntegerType = Optional[int]


class MeshBase(ABC):
    __slots__ = ['nodes', 'cells', 'facets']

    def __init__(self, **kwargs):
        self.nodes = self._impl_generate_nodes(**kwargs)
        self.cells = self._impl_generate_cells(**kwargs)
        self.facets = self._impl_generate_facets(**kwargs)

    def cell(self, index: ExIntegerType = None) -> List[IntegerType]:
        if index is None:
            return self.cells
        return self.cells[index]

    def node(self, index: ExIntegerType = None):
        if index is None:
            return self.nodes
        return self.nodes[index]

    def cell_coordinates(self, i: IntegerType):
        return self.nodes[self.cells[i]]

    def facet(self, i: IntegerType):
        return self.facets[i]

    def num_cells(self) -> int:
        return len(self.cells)

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_facets(self) -> int:
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
    def __init__(self, n: IntegerType, degree: IntegerType):
        super().__init__(n=n, degree=degree)

    def _impl_generate_nodes(self, **kwargs):
        n = kwargs['n'] if 'n' in kwargs else 0
        from numpy import linspace
        return linspace(0, 1, n + 1).reshape(-1, 1)

    def _impl_generate_cells(self, **kwargs):
        n = kwargs['n'] if 'n' in kwargs else 0
        return [[i + j for j in range(2)] for i in range(n)]

    def _impl_generate_facets(self, **kwargs):
        return []
