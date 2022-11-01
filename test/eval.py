import numpy as np


def qr(n=4):
    sqrt = np.sqrt
    asarray = np.asarray
    if n == 1:
        gp = asarray([0.0])
        gw = asarray([2.0])
    elif n == 2:
        p = sqrt(1/3)
        gp = asarray([p, -p])
        gw = asarray([1.0, 1.0])
    elif n == 3:
        gp = asarray([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
        gw = asarray([5.0, 8.0, 5.0]) / 9.0
    elif n == 4:
        p0 = sqrt(3/7-2/7*np.sqrt(6/5))
        p1 = sqrt(3/7+2/7*np.sqrt(6/5))
        gp = asarray([-p1, -p0, p0, p1])
        w0 = 0.5 + sqrt(30) / 36
        w1 = 0.5 - sqrt(30) / 36
        gw = asarray([w1, w0, w0, w1])
    elif n == 5:
        p0 = 1/3 * sqrt(5-2*sqrt(10/7))
        p1 = 1/3 * sqrt(5+2*sqrt(10/7))
        gp = asarray([-p1, -p0, 0.0, p0, p1])
        w0 = (322 + 13*sqrt(70))/900
        w1 = (322 - 13*sqrt(70))/900
        gw = asarray([w1, w0, 128/225, w0, w1])
    return (gp + 1) * 0.5, gw * 0.5


def BSpline(x):
    return (0 <= x) * (x < 1) * (1 - 2 * x) ** 2


def f(x):
    pi = np.pi
    return -pi**2 * np.sin(pi*x)


if __name__ == '__main__':
    from scipy.integrate import quad as fint
    q = fint(lambda x: f(x)*BSpline(x), 0, 0.5)
    gp, gw = qr(5)
    xg = gp * 0.5
    q2 = np.sum(f(xg) * BSpline(xg) * gw * 0.5)
    print(q, q2)
