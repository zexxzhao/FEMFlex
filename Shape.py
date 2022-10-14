from abc import ABC, abstractmethod

class ShapeFunctionBase(ABC):
    __slots__ = ['shape_functions', 'ready']

    def __init__(self, order=-1, **kwargs):
        if order == -1:
            self.shape_functions = [self.generate_bais_functions(i, **kwargs) for i in range(2)]
        elif order > 0:
            self.shape_functions = [self.generate_bais_functions(i, **kwargs) for i in range(order + 1)]

        self.ready = False
        self._impl_flush()

    def _impl_register(self, f, loc):
        if self.ready:
            raise RuntimeError('Registration in a ready Shape is not allowed.')
        if not isinstance(loc, (tuple, list)):
            raise TypeError(f"Expect a tuple or list, got a {type(loc)}.")
        if len(loc) != 2 or any([int(s) != s for s in loc]):
            raise ValueError(f"Expect two integers.")
        order, idx = loc[0], loc[1]
        self.shape_functions[order][idx] = f

    def _impl_flush(self):
        if self.ready:
            raise RuntimeError('Flushing in a ready Shape is not allowed.')
        n = len(self.shape_functions[0])
        if any([n != len(fn) for fn in self.shape_functions]):
            from warnings import warn
            warn('Not ready. Try again.', UserWarning, stacklevel=2)
        self.ready = True

    def _impl_reset(self):
        self.shape_functions = []
        self.ready = False
    
    def is_ready(self):
        return self.ready
    def internal_check(foo):
        def func_avail(*args, **kwargs):
            if not args[0].is_ready():
                raise RuntimeError("Not ready.")
            return foo(*args. **kwargs)
        return func_avail

    @internal_check
    def eval(self, **kwargs):
        order = kwargs['order']
        x = kwargs['x']
        basis = self.shape_functions[order]
        from numpy import asarray
        if len(kwargs) == 2:
            return asarray([f(x) for f in basis])
        elif len(kwargs) == 3:
            return asarray([basis[i](x) for i in kwargs['index']])
        else:
            raise ValueError("Unknown parameters.")

    def base(self, **kwargs):
        return self.eval(order=0, **kwargs)

    def first_derivative(self, **kwargs):
        return self.eval(order=1, **kwargs)
    
    @abstractmethod
    def generate_basis_functions(self, order, **kwargs):
        pass


class Shape1DIGA(ShapeFunctionBase):
    def __init__(self, order=2):
        super().__init__(-1)
        
    def generate_basis_functions(self, i):
        from scipy.interpolate import BSpline

