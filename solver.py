import numpy as np
from numba import cuda
from kdtree import *

Dz = 0.5
#Dz = 0.254 #https://up.codes/s/wall-thickness
nz = 10
dz = Dz / (nz - 1)
dt = 0.001
nt = 1
Dt = dt * nt
tint = 297.0

@cuda.jit((numba_tri_dt[:], int32, float32[:,:]))
def solverInit(tri, nz, T):
    i = cuda.grid(1)
    if i >= len(tri):
        return

    tout = tri[i]['T']
    tstp = (tout - tint) / (nz - 1)
    for j in range(nz + 2):
        T[i][j] = tint + tstp * (j - 1)

@cuda.jit((numba_tri_dt[:], float64[:,:], float32[:,:], float32[:,:]))
def solverStep(tri, E_permanent, T, d2Tdz2):
    i = cuda.grid(1)
    if i >= len(tri):
        return

    if tri[i]['A'] != 0.0:
        Rn = (E_permanent[i][0] + E_permanent[i][1] + E_permanent[i][2]) / tri[i]['A']
        Hs = 0.0
        G = Rn - Hs
        
        for s in range(nt):
            for j in range(2, nz):
                d2Tdz2[i][j] = (T[i][j + 1] - 2 * T[i][j] + T[i][j - 1]) / (dz ** 2)

            for j in range(2, nz):
                T[i][j] += dt * tri[i]['k'] * d2Tdz2[i][j]                         #gradient update
            
            T[i][1] = T[i][1]                                            #boundary condition
            T[i][0] = 2 * T[i][1] - T[i][2]                              #linearity
            T[i][nz + 1] = T[i][nz - 1] + 2 * dz * G / tri[i]['k']       #boundary condition
            T[i][nz] = (T[i][nz + 1] + T[i][nz - 1]) / 2                 #linearity
        
        tri[i]['T'] = T[i][nz]
        
