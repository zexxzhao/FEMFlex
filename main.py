from abc import ABC, abstractmethod
import numpy as np
import scipy
import matplotlib.pyplot as plt



class DOF:
    def __init__(self, mesh, k=2):
        self.mesh = mesh
        vtx = mesh.vertex()
        self.knot = np.hstack(([vertex[0]] * k, vertex, [vertex[-1]] * k))
        nknot = self.knot.shape[0]
        self.base = scipy.interpolate.BSpline(k, np.zeros((nkot - k - 1)), k)

    def 
        
