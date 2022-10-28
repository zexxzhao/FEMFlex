import numpy as np
import matplotlib.pyplot as plt

e = [0.00063707859, 0.0020569838707, 0.0075703609715]
x = 1/np.array([40, 20, 10])

plt.loglog(x, e, c='b', label='2nd iga')
#plt.text(1, 1, 'k=1.785', color='b')
p = np.polyfit(np.log(x), np.log(e), 1)
print(p)
plt.xlabel(r'$h$', fontsize=20)
plt.ylabel(r'$e$', fontsize=20)
#x = 1.0 / 2 ** np.arange(16)
eref = 1e-0 * x ** 2
plt.loglog(x, eref, c='r', label='quadratic')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
