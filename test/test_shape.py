# test shape.py

import femflex as flex


def _test_bspline(k, visual=False):
    spline = flex.Shape1DIGA(k)
    assert spline.get_num_basis_functions() == k + 1 + 2 * sum(range(k + 1))

    for order in range(k):
        val0 = spline.eval(0.5, order)
        assert val0.shape[0] == (k + 1) ** 2
        assert all([v is not None for v in val0])

    if visual:
        from numpy import linspace
        import matplotlib.pyplot as plt
        x = linspace(0, 1, 1000 + 1)
        v = spline.eval(x)
        plt.figure()
        for i in range(v.shape[0]):
            plt.plot(x, v[i], label=f'f[{i}]')
        plt.legend()
        plt.show()


def test_bspline():
    _test_bspline(0, 0)
    _test_bspline(1, 0)
    _test_bspline(2, 0)


if __name__ == '__main__':
    test_bspline()
