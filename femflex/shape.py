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
            if isinstance(index, Iterable):
                return asarray([basis(i)(x) for i in index])
            else: # is an integer
                return basis(index)(x)
        else:
            nbasis = self.get_num_basis_functions(kwargs)
            return asarray([basis(i)(x) for i in range(nbasis)])

    def base(self, **kwargs):
        return self.eval(order=0, **kwargs)

    def first_derivative(self, **kwargs):
        return self.eval(order=1, **kwargs)
    
    def nth_derivative(self, n, **kwargs):
        return self.eval(order=n, **kwargs) 

    @abstractmethod
    def get_basis_functions(self, order, index, **kwargs):
        pass

    @abstractmethod
    def get_num_basis_functions(self, **kwargs):
        pass

class BSpline:
    def __init__(self, x, c, k):
        from numpy import asarray
        self.x = asarray(x)
        self.c = asarray(c)
        self.k = k
    def __call__(self, x):
        return bspline(self.x, x, self.c, k)
    @staticmethod
    def B(x, k, i, t):
        if k == 0:
          return 1.0 if t[i] <= x < t[i+1] else 0.0
        if t[i+k] == t[i]:
          c1 = 0.0
        else:
          c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
        if t[i+k+1] == t[i+1]:
          c2 = 0.0
        else:
          c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
        return c1 + c2

    @staticmethod
    def bspline(x, t, c, k):
        n = len(t) - k - 1
        assert (n >= k+1) and (len(c) >= n)
        return sum(c[i] * B(x, k, i, t) for i in range(n))

class Shape1DIGA(ShapeFunctionBase):
    __slots__ = ['base_fn', 'porder']
    def __init__(self, porder=2):
        super().__init__() 
        self.porder = porder
        base = []
        #from scipy.interpolate import BSpline
        if porder == 2:
            base.append(BSpline([0, 0, 0, 1], [1], self.porder))
            base.append(BSpline([0, 0, 1, 2], [1], self.porder))
            base.append(BSpline([-1, -1, 0, 1], [1], self.porder))
            base.append(BSpline([0, 1, 2, 3], [1], self.porder))
            base.append(BSpline([-1, 0, 1, 2], [1], self.porder))
            base.append(BSpline([-2, -1, 0, 1], [1], self.porder))
            base.append(BSpline([-1, 0, 1, 1], [1], self.porder))
            base.append(BSpline([0, 1, 2, 2], [1], self.porder))
            base.append(BSpline([0, 1, 1, 1], [1], self.porder))
        else:
            raise ValueError("Not supported yet.")
        self.base_fn = base

    def get_num_basis_functions(self):
        p = self.porder
        if p == 1:
            return 2
        else:
            return (p + 2) * (p + 1) // 2

    def get_basis_functions(self, dorder, index):
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
