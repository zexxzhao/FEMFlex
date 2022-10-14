from abc import ABC, abstractmethod

class ShapeFunctionBase(ABC):
    def eval(self, **kwargs):
        dorder = kwargs['dorder']
        x = kwargs['x']

        def basis(index):
            return self.get_basis_function(dorder, index, kwargs)
        from numpy import asarray
        if 'index' in kawrgs:
            index = kwargs['index']
            from collections.abc import Iterable
            if isinstance(index, Iterable)
                return asarray([basis(i)(x) for i in index])
            else: # is an integer
                return basis(index)(x)
        else:
            nbasis = self.get_num_basis_functions(kwargs)
            return asarray([basis(i)(x) for i in range(nbasis)])

    def base(self, **kwargs):
        return self.eval(order=0, kwargs)

    def first_derivative(self, **kwargs):
        return self.eval(order=1, kwargs)
    
    def nth_derivative(self, n, **kwargs):
        return self.eval(order=n, kwargs) 

    @abstractmethod
    def get_basis_functions(self, order, index, **kwargs):
        pass

    @abstractmethod
    def get_num_basis_functions(self, **kwargs):
        pass

class Shape1DIGA(ShapeFunctionBase):
    __slots__ = ['base_fn', 'porder']
    def __init__(self, porder=2):
        super().__init__() 
        self.porder = porder
        base = []
        base.append(BSpline([0, 0, 0, 1], [1], self.porder))
        base.append(BSpline([0, 0, 1, 2], [1], self.porder))
        base.append(BSpline([-1, -1, 0, 1], [1], self.porder))
        base.append(BSpline([0, 1, 2, 3], [1], self.porder))
        base.append(BSpline([-1, 0, 1, 2], [1], self.porder))
        base.append(BSpline([-2, -1, 0, 1], [1], self.porder))
        self.base_fn = base
    def get_num_basis_functions(self):
        if self.porder == 1:
            return 2
        else:
            return -1
    def get_basis_functions(self, dorder, index):
        from scipy.interpolate import BSpline
        def derivative(e, n):
            for _ in range(n):
                e = e.derivative()
            return e
        if dorder == 0:
            return [e.base_element() for e in self.base]
        elif dorder == 1:
            return [e.derivative() for e in self.base]
        else:
            return [derivative(e, dorder) for e in self.base]
