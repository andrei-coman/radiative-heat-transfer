from kdtree import *
from numba import cuda

'''
CUDA kernel that transfers the energy of a ray
@param {numba_tri_dt[:]} tri - array of triangles in format numba_tri_dt (kdtree.py)
@param {numba_small_ray_dt[:]} rays - array of rays in format numba_small_ray_dt[:] (kdtree.py)
@param {float64[:]} E_permanent - for each triangle tri[i], E_permanent[i] is the energy "permanently" accumulated by the triangle
@param {float64[:]} E_temporary - for each triangle tri[i], E_temporary[i] is the energy that needs to get emitted by the triangle
'''
@cuda.jit((numba_tri_dt[:], numba_small_ray_dt[:], float64[:,:], float64[:,:], float64[:]))
def addPath(tri, rays, E_permanent, E_temporary, fsum):
    i = cuda.grid(1) #thread index
    if i >= len(rays):
        return
    
    ray = rays[i] #current ray that we are processing
    for band in range(3):
        E = ray['E'][band] * tri[ray['t']]['E_to_emit'][band] * fsum[band] #the energy of the current ray
        cuda.atomic.add(E_permanent, (ray['s'], band),      tri[ray['s']]['alpha'][band]  * E) #part of the energy is permanently retained
        cuda.atomic.add(E_temporary, (ray['s'], band), (1 - tri[ray['s']]['alpha'][band]) * E) #part of the energy will be re-emitted in the next reflection step

'''
CUDA kernel that transfers the energy of a ray
@param {numba_tri_dt[:]} tri - array of triangles in format numba_tri_dt (kdtree.py)
@param {numba_tiny_ray_dt[:]} rays - array of rays in format numba_tiny_ray_dt[:] (kdtree.py)
@param {float64[:]} E_permanent - for each triangle tri[i], E_permanent[i] is the energy "permanently" accumulated by the triangle
@param {float64[:]} E_temporary - for each triangle tri[i], E_temporary[i] is the energy that needs to get emitted by the triangle
'''
@cuda.jit((numba_tri_dt[:], numba_tiny_ray_dt[:], float64[:,:], float64[:,:]))
def addDiffusePath(tri, rays, E_permanent, E_temporary):
    i = cuda.grid(1)
    if i >= len(rays):
        return
    
    ray = rays[i]
    for band in range(3):
        E = ray['E'][band]
        cuda.atomic.add(E_permanent, (ray['t'], band),      tri[ray['t']]['alpha'][band]  * E) #part of the energy is permanently retained
        cuda.atomic.add(E_temporary, (ray['t'], band), (1 - tri[ray['t']]['alpha'][band]) * E) #part of the energy will be re-emitted in the next reflection step

'''
CUDA kernel that traces a ray with specular reflection and transfers its energy accordingly
@param {numba_tri_dt[:]} tri - array of triangles in format numba_tri_dt (kdtree.py)
@param {int32} root - index of the root-node of the kdtree in the kdnodes array
@param {numba_ray_dt} ray - current ray to be traced
@param {float64} E - energy of ray
@param {int32} i - index of ray
@param {float64[:,:]} bounds - bounding rectangle for the geometry
@param {numba_stack_dt[:,:]} stacks - memory reserved for the kdsearch stacks
@param {numba_kdnode_dt[:]} kdnodes - array of nodes of the kdtree
@param {int32[:]} kdtriangles - array that indicates which triangles correspond to each kdnode
@param {numba_seg_dt[:,:]} segments - array of segments (ray fragments) to be plotted
@param {int32} MAX_REFLECTIONS - number of reflections that the ray should be followed for
@param {float64[:]} E_permanent - for each triangle tri[i], E_permanent[i] is the energy "permanently" accumulated by the triangle
@param {float64[:]} E_temporary - for each triangle tri[i], E_temporary[i] is the energy that needs to get emitted by the triangle
'''
@cuda.jit(device = True)
def TraceRay(tri, root, ray, E, i, bounds, stacks, kdnodes, kdtriangles, segments, MAX_REFLECTIONS, E_permanent, E_temporary):
    mint, which = INFINITY, 0
    num = 0 #number of reflections so far

    while which != -1 and num <= MAX_REFLECTIONS:
        mint, which = kdSearch(tri, i, root, bounds, ray, kdnodes, kdtriangles, stacks) #t parameter of the ray-triangle intersection; which triangle is intersected

        for j in range(3): #add to the array of segments
            segments[i][num]['which'] = which
            segments[i][num]['a'][j] = ray['p'][j]
            segments[i][num]['b'][j] = ray['p'][j] + mint * ray['r'][j]

        if which != -1:
            for band in range(3):
                cuda.atomic.add(E_permanent, (which, band), tri[which]['alpha'][band] * E[band]) #transfer energy to triangle as "permanent"
                E[band] *= (1 - tri[which]['alpha'][band]) #update ray energy
            
        num += 1
        if num <= MAX_REFLECTIONS:
            reflection(ray, mint, tri[which])

    if which != -1:
        for band in range(3):
            cuda.atomic.add(E_temporary, (which, band), E[band]) #keep the remaining energy in the last triangle, to be re-emitted

'''
CUDA kernel that traces a ray with specular reflection and transfers its energy accordingly
@param {numba_tri_dt[:]} tri - array of triangles in format numba_tri_dt (kdtree.py)
@param {int32} root - index of the root-node of the kdtree in the kdnodes array
@param {numba_ray_dt[:]} rays - array of rays in format numba_ray_dt[:] (kdtree.py)
@param {float64[:,:]} bounds - bounding rectangle for the geometry
@param {numba_stack_dt[:,:]} stacks - memory reserved for the kdsearch stacks
@param {numba_kdnode_dt[:]} kdnodes - array of nodes of the kdtree
@param {int32[:]} kdtriangles - array that indicates which triangles correspond to each kdnode
@param {numba_seg_dt[:,:]} segments - array of segments (ray fragments) to be plotted
@param {int32} MAX_REFLECTIONS - number of reflections that the ray should be followed for
@param {float64[:]} E_permanent - for each triangle tri[i], E_permanent[i] is the energy "permanently" accumulated by the triangle
@param {float64[:]} E_temporary - for each triangle tri[i], E_temporary[i] is the energy that needs to get emitted by the triangle
'''
@cuda.jit((numba_tri_dt[:], int32, numba_ray_dt[:], float64[:,:], numba_stack_dt[:,:], numba_kdnode_dt[:], int32[:], numba_seg_dt[:,:], int32, float64[:,:], float64[:,:]))
def addRayKD2(tri, root, rays, bounds, stacks, kdnodes, kdtriangles, segments, MAX_REFLECTIONS, E_permanent, E_temporary):
    i = cuda.grid(1) #thread index
    if i >= len(rays):
        return
    ray = rays[i] #current ray we are processing

    PAR = ray['E'][0]
    NIR = ray['E'][1]
    L   = ray['E'][2]

    for band in range(3):
        if math.isnan(ray['E'][band]): #if invalid ray
            return
    TraceRay(tri, root, ray, ray['E'], i, bounds, stacks, kdnodes, kdtriangles, segments, MAX_REFLECTIONS, E_permanent, E_temporary)

    ray['E'][0] = PAR
    ray['E'][1] = NIR
    ray['E'][2] = L
'''
CUDA kernel that traces a ray with specular reflection and transfers its energy accordingly
@param {numba_tri_dt[:]} tri - array of triangles in format numba_tri_dt (kdtree.py)
@param {int32} root - index of the root-node of the kdtree in the kdnodes array
@param {numba_big_ray_dt[:]} rays - array of rays in format numba_big_ray_dt[:] (kdtree.py)
@param {float64[:,:]} bounds - bounding rectangle for the geometry
@param {numba_stack_dt[:,:]} stacks - memory reserved for the kdsearch stacks
@param {numba_kdnode_dt[:]} kdnodes - array of nodes of the kdtree
@param {int32[:]} kdtriangles - array that indicates which triangles correspond to each kdnode
@param {numba_seg_dt[:,:]} segments - array of segments (ray fragments) to be plotted
@param {int32} MAX_REFLECTIONS - number of reflections that the ray should be followed for
@param {float64[:]} E_permanent - for each triangle tri[i], E_permanent[i] is the energy "permanently" accumulated by the triangle
@param {float64[:]} E_temporary - for each triangle tri[i], E_temporary[i] is the energy that needs to get emitted by the triangle
'''
@cuda.jit((numba_tri_dt[:], int32, numba_big_ray_dt[:], float64[:,:], numba_stack_dt[:,:], numba_kdnode_dt[:], int32[:], numba_seg_dt[:,:], int32, float64[:,:], float64[:,:]))
def addRayKD1(tri, root, rays, bounds, stacks, kdnodes, kdtriangles, segments, MAX_REFLECTIONS, E_permanent, E_temporary):
    i = cuda.grid(1) #thread index
    if i >= len(rays):
        return
    ray = rays[i] #current ray we are processing

    PAR = ray['E'][0]
    NIR = ray['E'][1]
    L   = ray['E'][2]

    for band in range(3):
        ray['E'][band] *= tri[ray['index']]['E_to_emit'][band]
        if math.isnan(ray['E'][band]): #if invalid ray
            return
    TraceRay(tri, root, ray, ray['E'], i, bounds, stacks, kdnodes, kdtriangles, segments, MAX_REFLECTIONS, E_permanent, E_temporary)

    ray['E'][0] = PAR
    ray['E'][1] = NIR
    ray['E'][2] = L

