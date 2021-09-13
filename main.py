import STL
import random
import math

import matplotlib.pyplot as plt

from solver import *
from launcher import *
from renderer import *
from kdtree import *
from creator import *
from BuildingSegmenter import *

from numba import cuda

STANDARD_TEMP = 300

STANDARD_EMISSIVITY_PAR = 0.0
STANDARD_EMISSIVITY_NIR = 0.0
STANDARD_EMISSIVITY_L   = 0.85

STANDARD_ABSORPTIVITY_PAR = 0.70
STANDARD_ABSORPTIVITY_NIR = 0.70
STANDARD_ABSORPTIVITY_L   = 0.70

STEFAN_BOLTZMANN = 5.67E-8

def BuildTriangles(tri, k):
    trivec = list()
    for T in tri:
        u, v, w = T.vertices
        u = np.array(u)
        v = np.array(v)
        w = np.array(w)

        n = np.array(T.n)
        x, y = None, None
        if n[2] != 0:
            x = np.array([0, n[2], -n[1]])
        else:
            x = np.array([n[1], -n[0], 0])
        x /= nla.norm(x)
        y = np.cross(n, x)

        A = nla.norm(np.cross(u - w, v - w)) / 2.0
        d = n.dot((u + v + w) / 3.0)

        T = STANDARD_TEMP
        eps   = [STANDARD_EMISSIVITY_PAR, 
                 STANDARD_EMISSIVITY_NIR, 
                 STANDARD_EMISSIVITY_L]
        alpha = [STANDARD_ABSORPTIVITY_PAR, 
                 STANDARD_ABSORPTIVITY_NIR, 
                 STANDARD_ABSORPTIVITY_L]
        E_to_emit = [0.0, 0.0, 0.0]

        trivec.append((u, v, w, n, x, y, d, A, T, E_to_emit, eps, alpha, k))
    trivec = np.array(trivec, dtype = numba_tri_dt)
    return trivec

def PrepareEnergy(tri, E_permanent, E_temporary):
    for i in range(len(tri)):
        for band in range(3):
            tri[i]['E_to_emit'][band] = tri[i]['eps'][band] * STEFAN_BOLTZMANN * (tri[i]['T'] ** 4) * tri[i]['A']
            E_permanent[i][band] = -tri[i]['E_to_emit'][band]
            E_temporary[i][band] = 0

@cuda.jit
def TransformTriangles(tri):
    i = cuda.grid(1)
    if i >= len(tri):
        return
    for j in range(3):
        tri[i]['v'][j] -= tri[i]['u'][j]
        tri[i]['w'][j] -= tri[i]['u'][j]

if __name__ == "__main__":
    initializeRenderer()

    #STL.lazToStl("../data/BUILDINGS.laz", "../data/BUILDINGS.stl", "../data/GROUND.stl", BUILDING_RESOLUTION, GROUND_RESOLUTION)
    faces = STL.ReadSTL("../data/BUILDINGS.stl")
    ground = STL.ReadSTL("../data/GROUND.stl")
    
    triFaces = BuildTriangles(faces, 1.70)
    triGround = BuildTriangles(ground, 1.70)
    tri = np.concatenate((triFaces, triGround), axis = 0)
    if len(tri) > MAX_TRI:
        print("Too many triangles. Change parameters.")
        sys.exit()

    E_permanent = np.zeros((len(tri),3), dtype = np.float64)
    E_temporary = np.zeros((len(tri),3), dtype = np.float64)
    
    #BuildTree(tri, 'kdtree.npy')
    TransformTriangles[THREADS_PER_BLOCK_TRI, NUM_BLOCKS](tri)
    bounds, kdnodes, kdtriangles, root, depth = LoadKdTree('kdtree.npy')
    batch_segments = np.array([[seg_sample] * (MAX_REFLECTIONS + 1)] * BATCH_SIZE, dtype = numba_seg_dt)
    stacks = np.array([[stack_sample] * depth] * BATCH_SIZE, dtype = numba_stack_dt)

    view = CreateAllRays(tri, bounds, kdnodes, kdtriangles, root, depth, E_permanent, E_temporary, recompute = False)

    frc = list()
    frcR = list()
    with open('frc.npy', 'rb') as f:
        frc = np.load(f)
    with open('frcR.npy', 'rb') as f:
        frcR = np.load(f)

    E_permanent = np.zeros((len(tri),3), dtype = np.float64)
    E_temporary = np.zeros((len(tri),3), dtype = np.float64)
    T           = np.zeros((len(tri), nz + 2), dtype = np.float32)
    d2Tdz2      = np.zeros((len(tri), nz + 2), dtype = np.float32) 
    
    solverInit[THREADS_PER_BLOCK_TRI, NUM_BLOCKS](tri, nz, T)
    for step in range(5):
        PrepareEnergy(tri, E_permanent, E_temporary)
        
        segments1 = LaunchRaysFromFile('direct_sun.npy', tri, 
                                       bounds, kdnodes, kdtriangles, root, depth, 
                                       range(35212, 35242), False, 0, 
                                       E_permanent, E_temporary, batch_segments, stacks)

        DiffuseRaysFromFile('diffuse_path.npy', tri, E_permanent, E_temporary)
        ReflectRaysFromFile('emitted_path.npy', tri, E_permanent, E_temporary, 11, True, frc, frcR)

        solverStep[THREADS_PER_BLOCK_TRI, NUM_BLOCKS](tri, E_permanent, T, d2Tdz2)
        
        maxT = 0.0
        minT = 1000000.0
        bestm = 0
        bestM = 0
        for i in range(len(tri)):
            if tri[i]['T'] > maxT:
                maxT = tri[i]['T']
                bestM = i
            if tri[i]['T'] < minT:
                minT = tri[i]['T']
                bestm = i

        print(maxT, bestM, E_permanent[bestM][0] + E_permanent[bestM][1] + E_permanent[bestM][2], tri[bestM]['A'], (E_permanent[bestM][0] + E_permanent[bestM][1] + E_permanent[bestM][2]) / tri[bestM]['A'])
        print(minT, bestm, E_permanent[bestm][0] + E_permanent[bestm][1] + E_permanent[bestm][2], tri[bestm]['A'], (E_permanent[bestm][0] + E_permanent[bestm][1] + E_permanent[bestm][2]) / tri[bestm]['A'])
        
    offset = len(faces)
    Tmin = INFINITY
    Tmax = -INFINITY
    for i in range(len(tri)):
        Tmin = min(Tmin, tri[i]['T'])
        Tmax = max(Tmax, tri[i]['T'])

    #faces = STL.toArray(faces, tri, 0, lambda n, T, i, Tmin=Tmin, Tmax=Tmax, view=view: addNoise(darken(normalColoring(n, Tmin, Tmax, T), Tmin, Tmax, T), Tmin, Tmax, T) * view[i])
    #ground = STL.toArray(ground, tri, offset, lambda n, T, i, Tmin=Tmin, Tmax=Tmax, view=view: addNoise(groundColoring(n, Tmin, Tmax, T), Tmin, Tmax, T) * view[i])
    Tmax = 350
    faces = STL.toArray(faces, tri, 0, lambda n, T, i, Tmin=Tmin, Tmax=Tmax, view=view: tempColoring(n, Tmin, Tmax, T))
    ground = STL.toArray(ground, tri, offset, lambda n, T, i, Tmin=Tmin, Tmax=Tmax, view=view: tempColoring(n, Tmin, Tmax, T))
    triangles = faces + ground
    bufferTri = CreateBuffer(triangles)
    
    BS = BuildingSegmenter()
    BS.LoadPoints("../data/TREES.laz")
    BS.CreateLeaves()
    lArray = BS.getLeaves()
    bufferLvs = CreateBuffer(lArray)

    '''
    dx = 1024
    dy = 1024
    dz = 1024
    cubes = {}
    noLeaves = len(lArray) // 6
    for i in range(noLeaves):
        coords = lArray[6 * i:6 * i + 3]
        cube = (coords[0] // dx, coords[1] // dy, coords[2] // dz)
        if cube not in cubes:
            cubes[cube] = 0
        cubes[cube] += 1
    '''
    
    glRotatef(60.0, 0, 1, 0)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cudaProfilerStop()
                pygame.quit()
                quit()

        glRotatef(0.25, 0, 1, 0)
        render(bufferTri, triangles, bufferLvs, lArray)
        
        for i in range(len(segments1)):
            for entry in segments1[i]:
                p = entry['a']
                q = entry['b']
                plotSegment(p, q, 0.0, 1.0, 0.0)
                if entry['which'] == -1:
                    break
    
        pygame.display.flip()
        pygame.time.wait(40)
    
