from abc import ABC, abstractmethod


class ShapeFunctionBase(ABC):
    def eval(self, x, order=0, index=None):
        if index is None:
            index = range(self.get_num_basis_functions())
        basis_fn = self.get_basis_functions(order, index)
        from numpy import asarray
        return asarray([e(x) for e in basis_fn])

    def base(self, **kwargs):
        return self.eval(order=0, **kwargs)

    def first_derivative(self, **kwargs):
        return self.eval(order=1, **kwargs)

    def nth_derivative(self, n, **kwargs):
        return self.eval(order=n, **kwargs)

    @abstractmethod
    def get_basis_functions(self, order, index):
        pass

    @abstractmethod
    def get_num_basis_functions(self, **kwargs):
        pass


class Shape1DIGA(ShapeFunctionBase):
    __slots__ = ['base_fn', 'degree']

    def __init__(self, k=2):
        super().__init__()
        self.degree = k
        from numpy import asarray

        def bspline_knots(k):
            return [0] * k + [i for i in range(k + 2)] + [k + 1] * k
        knots = asarray(bspline_knots(k))

        base = []
        from scipy.interpolate import BSpline

        num_full_basis = 2 * k + 1
        for i in range(num_full_basis):
            local_knots = knots[i:i+k+2]
            knot_range = max(local_knots) - min(local_knots)
            coef = [0] * num_full_basis
            coef[i] = 1
            for j in range(knot_range):
                base.append(BSpline(knots - j - min(local_knots),
                                    coef, k, extrapolate=False))

        self.base_fn = base

    def get_num_basis_functions(self, degree=None):
        if degree is None:
            p = self.degree
        else:
            p = degree
        return (p + 1) ** 2

    def get_basis_functions(self, order, index):
        def derivative(f, d):
            def f1(x, h=1e-6):
                return (f(x+h) - f(x-h)) / (2.0*h)
            return f if d == 0 else f1
        from collections.abc import Sequence
        if isinstance(index, Sequence):
            return [derivative(self.base_fn[idx], order) for idx in index]
        else:
            return derivative(self.base_fn[index], order)
