import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(11, 4.5))
plt.subplot(121)
dat = np.loadtxt('error_fenics.txt')
x = 1 / 2 ** np.arange(1, 1 + dat.shape[0])
e = dat[:, 0]
plt.loglog(x, e, c='m', label='FEM')
# plt.text(1, 1, 'k=1.785', color='b')
p = np.polyfit(np.log(x), np.log(e), 1)
print(p)

dat = np.loadtxt('error.txt')
x = 1 / 2 ** np.arange(1, 1 + dat.shape[0])
e = dat[:, 0]
plt.loglog(x, e, c='b', label='IGA L2')
p = np.polyfit(np.log(x[-4:]), np.log(e[-4:]), 1)
print(p)

plt.xlabel(r'$h$', fontsize=20)
plt.ylabel(r'$e$', fontsize=20)
eref = 3e-1 * x ** 2
plt.loglog(x, eref, '--r', label='k=2')
eref = 5e-1 * x ** 2.5
plt.loglog(x, eref, ls='--', c='Tab:blue', label='k=2.5')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=12)
plt.title('Poisson: 2nd IGA vs 1st FEM', fontsize=15)
plt.tight_layout()

plt.subplot(122)
dat = np.loadtxt('error_fenics.txt')
x = 1 / 2 ** np.arange(1, 1 + dat.shape[0])
e = dat[:, 0]
plt.loglog(x, e, c='b', label='FEM linear')
# plt.text(1, 1, 'k=1.785', color='b')
p = np.polyfit(np.log(x), np.log(e), 1)
print(p)

dat = np.loadtxt('error_fenics2.txt')
x = 1 / 2 ** np.arange(1, 1 + dat.shape[0])
e = dat[:, 0]
plt.loglog(x, e, c='r', label='FEM quadratic')
# plt.text(1, 1, 'k=1.785', color='b')
p = np.polyfit(np.log(x), np.log(e), 1)
print(p)

plt.xlabel(r'$h$', fontsize=20)
plt.ylabel(r'$e$', fontsize=20)
eref = 3e-1 * x ** 2
plt.loglog(x, eref, '--b', label='k=2')
eref = 5e-1 * x ** 3
plt.loglog(x, eref, '--r', label='k=3')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=12)
plt.title('Heat conduction: 2nd FEM vs 1st FEM', fontsize=15)

plt.tight_layout()
plt.show()
