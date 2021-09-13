import math
import datetime
from numba import cuda

from tracer import *
from renderer import *
from kdtree import *

THREADS_PER_BLOCK     = 4096
THREADS_PER_BLOCK_TRI = 4096
THREADS_PER_BLOCK_RAY = 3
NUM_BLOCKS            = 200
BATCH_SIZE            = THREADS_PER_BLOCK * NUM_BLOCKS
MAX_TRI               = THREADS_PER_BLOCK_TRI * NUM_BLOCKS  
MAX_RAY               = THREADS_PER_BLOCK_RAY * NUM_BLOCKS

'''
CUDA kernel that converts "temporary" absorbed energy into energy to be emitted
@param {numba_tri_dt[:]} tri - array of triangles in format numba_tri_dt (kdtree.py)
@param {float64[:]} E_temporary - for each triangle tri[i], E_temporary[i] is the energy that needs to get emitted by the triangle
'''
@cuda.jit((numba_tri_dt[:], float64[:,:], float64[:,:], float64[:], float64[:,:], float64[:,:]))
def ResetEnergy(tri, E_temporary, tmp, frc, tmpR, frcR):
    i = cuda.grid(1)
    if i >= len(tri):
        return
    j = cuda.blockIdx.x
    for band in range(3):
        tri[i]['E_to_emit'][band] = E_temporary[i][band]
        cuda.atomic.add(tmp , (j, band), tri[i]['E_to_emit'][band] * frc[i] )
        cuda.atomic.add(tmpR, (j, band), tri[i]['E_to_emit'][band] * frcR[i][band])
        E_temporary[i][band] = 0

@cuda.jit((numba_tri_dt[:], float64[:,:]))
def ResetEnergy2(tri, E_temporary):
    i = cuda.grid(1)
    if i >= len(tri):
        return
    for band in range(3):
        tri[i]['E_to_emit'][band] += E_temporary[i][band]
        E_temporary[i][band] = 0

'''
Traces NUM_REFLECTIONS steps of the reflection. All rays (from all triangles) are traced simultaneously
@param {string} path - path to the rays (saved as source triangle, target triangle, fraction of energy of the source corresponding to this ray)
@param {numba_tri_dt[:]} tri - array of triangles in format numba_tri_dt (kdtree.py)
@param {float64[:]} E_permanent - for each triangle tri[i], E_permanent[i] is the energy "permanently" accumulated by the triangle
@param {float64[:]} E_temporary - for each triangle tri[i], E_temporary[i] is the energy that needs to get emitted by the triangle
@param {int32} NUM_REFLECTIONS - number of reflection steps to be traced
'''
def ReflectRaysFromFile(path, tri, E_permanent, E_temporary, NUM_REFLECTIONS, finish, frc, frcR):
    print("Tracing reflections")
    f = open(path, 'rb')
    a = datetime.datetime.now()

    ResetEnergy2[THREADS_PER_BLOCK_TRI, NUM_BLOCKS](tri, E_temporary)
    fsum = np.zeros((3,), dtype = np.float64)
    isum = np.zeros((3,), dtype = np.float64)

    tmp  = np.zeros((THREADS_PER_BLOCK_TRI, 3), dtype = np.float64)
    tmpR = np.zeros((THREADS_PER_BLOCK_TRI, 3), dtype = np.float64)
    for i in range(len(tri)):
        fsum += tri[i]['E_to_emit'] * frc[i]
        isum += tri[i]['E_to_emit'] * frcR[i]
    
    for reflection in range(NUM_REFLECTIONS): #for each step
        '''
        s = 0
        [s := s + tri[i]['E_to_emit'][band] + E_permanent[i][band] + E_temporary[i][band] for i in range(len(tri)) for band in range(3)]
        print(s)
        '''        

        fsum = np.array([fsum[band] / isum[band] if isum[band] != 0.0 else 0.0 for band in range(3)])
        f.seek(0)
        batch_rays = np.load(f)
        while len(batch_rays):
            addPath[THREADS_PER_BLOCK, NUM_BLOCKS](tri, batch_rays, E_permanent, E_temporary, fsum) #trace the rays
            batch_rays = np.load(f)

        print("Finished reflection", reflection + 1)
        
        fsum = np.zeros((3,), dtype = np.float64)
        isum = np.zeros((3,), dtype = np.float64)
        tmp  = np.zeros((THREADS_PER_BLOCK_TRI, 3), dtype = np.float64)
        tmpR = np.zeros((THREADS_PER_BLOCK_TRI, 3), dtype = np.float64)
        ResetEnergy[THREADS_PER_BLOCK_TRI, NUM_BLOCKS](tri, E_temporary, tmp, frc, tmpR, frcR)
        for i in range(THREADS_PER_BLOCK_TRI):
            fsum += tmp[i]
            isum += tmpR[i]

    b = datetime.datetime.now()
    c = b - a
    print("Finished tracing reflections in", c)

    if finish:
        s = 0.0
        for i in range(len(tri)): #remaining temporary energy is kept in the triangle
            for band in range(3):
                E_permanent[i][band] += tri[i]['E_to_emit'][band] + E_temporary[i][band]
                E_temporary[i][band] = 0
                tri[i]['E_to_emit'][band] = 0
                s += E_permanent[i][band]
        print(s)

    f.close()

'''
Traces incoming solar diffuse rays
@param {string} path - path to the rays (saved as source triangle, target triangle, fraction of energy of the source corresponding to this ray)
@param {numba_tri_dt[:]} tri - array of triangles in format numba_tri_dt (kdtree.py)
@param {float64[:]} E_permanent - for each triangle tri[i], E_permanent[i] is the energy "permanently" accumulated by the triangle
@param {float64[:]} E_temporary - for each triangle tri[i], E_temporary[i] is the energy that needs to get emitted by the triangle
'''
def DiffuseRaysFromFile(path, tri, E_permanent, E_temporary):
    f = open(path, 'rb')
    a = datetime.datetime.now()
    
    batch_rays = np.load(f)
    while len(batch_rays):
        addDiffusePath[THREADS_PER_BLOCK_TRI, NUM_BLOCKS](tri, batch_rays, E_permanent, E_temporary)
        batch_rays = np.load(f)

    b = datetime.datetime.now()
    c = b - a
    print("Finished tracing reflections in", c)
    f.close()

'''
Launches rays from a file and traces them for a given number of specular reflections
@param {string} path - path to the rays
@param {numba_tri_dt[:]} tri - array of triangles in format numba_tri_dt (kdtree.py)
@param {float64[:,:]} bounds - bounding rectangle for the geometry
@param {numba_kdnode_dt[:]} kdnodes - array of nodes of the kdtree
@param {int32[:]} kdtriangles - array that indicates which triangles correspond to each kdnode
@param {int32} root - index of the root-node of the kdtree in the kdnodes array
@param {int32} depth - depth of the kdtree
@param {int32[:]} rg - list of rays that need to be plotted (by index)
@param {bool} fromTriangle - 1 if rays are emitted by triangles, 0 otherwise
@param {int32} MAX_REFLECTIONS - number of specular reflection steps to be traced
@param {float64[:]} E_permanent - for each triangle tri[i], E_permanent[i] is the energy "permanently" accumulated by the triangle
@param {float64[:]} E_temporary - for each triangle tri[i], E_temporary[i] is the energy that needs to get emitted by the triangle
@param {numba_seg_dt[:,:]} batch_segments - array of segments (ray fragments) to be plotted
@param {numba_stack_dt[:,:]} stacks - memory reserved for the kdsearch stacks
'''
def LaunchRaysFromFile(path, tri, bounds, kdnodes, kdtriangles, root, depth, rg, fromTriangle, MAX_REFLECTIONS, E_permanent, E_temporary, batch_segments, stacks):
    f = open(path, 'rb')

    segments = list()
    kdnodes_global = cuda.to_device(kdnodes)
    kdtriangles_global = cuda.to_device(kdtriangles)
    stacks_global = cuda.to_device(stacks)

    print("Tracing rays")
    a = datetime.datetime.now()
    nextInRg, i = 0, 0
    batch_rays = np.load(f)
    while len(batch_rays):
        start = BATCH_SIZE * i
        end   = BATCH_SIZE * i + min(len(batch_rays), BATCH_SIZE)

        if fromTriangle:
            addRayKD1[THREADS_PER_BLOCK, NUM_BLOCKS](tri, root, batch_rays, bounds, stacks_global, kdnodes_global, kdtriangles_global, batch_segments, MAX_REFLECTIONS, E_permanent, E_temporary)
        else:
            addRayKD2[THREADS_PER_BLOCK, NUM_BLOCKS](tri, root, batch_rays, bounds, stacks_global, kdnodes_global, kdtriangles_global, batch_segments, MAX_REFLECTIONS, E_permanent, E_temporary)

        while nextInRg < len(rg) and start <= rg[nextInRg] < end:
            segments.append(copy.deepcopy(batch_segments[rg[nextInRg] - start]))
            nextInRg += 1

        batch_rays = np.load(f)
        i += 1
        
    b = datetime.datetime.now()
    c = b - a
    print("Finished tracing in", c)
    
    f.close()
    segments = np.array(segments, dtype = numba_seg_dt)
    return segments

