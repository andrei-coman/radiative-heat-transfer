import STL
import math
import datetime
import sys
import os
import numpy.linalg as nla

from tracer import *
from launcher import *
from renderer import *
from kdtree import *
from BuildingSegmenter import *

from numba import cuda

STEFAN_BOLTZMANN = 5.67E-8
S_PAR_DIR  = 600 #perp to sun_dir
S_NIR_DIR  = 600 #perp to sun_dir
S_PAR_DIFF = 100
S_NIR_DIFF = 100
S_L_DIFF   = 600

BUILDING_RESOLUTION = 1000.0
GROUND_RESOLUTION = 1000.0
MAX_REFLECTIONS = 12

def CreateAllRays(tri, bounds, kdnodes, kdtriangles, root, depth, E_permanent, E_temporary, recompute = False):
    RAYS_PER_PATCH = 500
    VARIATIONS = 64
    sun_dir = np.array([-2.0, -1.0, 1.0])

    view = FindSunRays(sun_dir, tri, bounds, kdnodes, kdtriangles, root, depth, E_permanent, E_temporary)
    if recompute:
        #CreateSunRays(sun_dir, tri, view, 'direct_sun.npy')
        #CreateEmittedRays(tri, RAYS_PER_PATCH, 'emitted.npy', 'prediffuse.npy', sun_dir, VARIATIONS)
        CreateEmittedPath('emitted.npy', 'emitted_path.npy', 'frc.npy', 'frcR.npy', tri, bounds, kdnodes, kdtriangles, root, depth, E_permanent, E_temporary, RAYS_PER_PATCH)
        CreateDiffuseRays('prediffuse.npy', 'diffuse.npy', 'diffuse_path.npy', tri, bounds, kdnodes, kdtriangles, root, depth, E_permanent, E_temporary)
    
        os.remove('prediffuse.npy')
        os.remove('diffuse.npy')
        os.remove('emitted.npy')

    return view

'''
Creares a list of M rays spread uniformly on a hemisfere
https://perswww.kuleuven.be/~u0017946/publications/Papers97/art97a-Saff-Kuijlaars-MI/Saff-Kuijlaars-MathIntel97.pdf

@param {number} N = 2M + 1 (twice the number of desired rays plus one)
@returns {list} V - the list of rays
                    each entry of the list is a tuple of the form (cos(Theta), sin(Theta), sin(Phi), cos(Phi))
                    where Theta, Phi are as mentioned in the paper
'''
def GenerateVectors(N, sun_dir, VARIATIONS):
    V = list()
    sqrtN = math.sqrt(N)
    
    for vrt in range(VARIATIONS):
        W = list()
        phi = 0
        for k in range(2, N):
            h = -1 + 2 * (k - 1) / (N - 1)
            phi += 3.6 / sqrtN * 1.0 / math.sqrt(1 - h**2)
            if h > 0:
                W.append((h, math.sqrt(1 - h**2), math.sin(phi + vrt), math.cos(phi + vrt)))
        W.append((1.0, 0.0, 0.0, 1.0))
        V.append(copy.deepcopy(W))
    
    x = np.array([1, 0, 0], dtype = np.float64)
    y = np.array([0, 1, 0], dtype = np.float64)
    z = np.array([0, 0, 1], dtype = np.float64)

    sun_dir /= np.linalg.norm(sun_dir)
    cosThetaStar = -sun_dir[1]
    secThetaStar = inv(cosThetaStar)
    sinThetaStar = math.sqrt(1 - cosThetaStar ** 2)
    Phi1 = math.atan2(-sun_dir[2], -sun_dir[0])

    rays = list()

    sum_S_DIFF = 0.0
    sum_L_DIFF = 0.0
    for h, hprime, sinPhi, cosPhi in V[0]:
        ray_dir = h * y + hprime * (sinPhi * z + cosPhi * x)
        dir_inv = np.array([inv(ray_dir[i]) for i in range(3)])

        cosTheta = ray_dir[1]
        secTheta = inv(cosTheta)
        sinTheta = math.sqrt(1 - cosTheta ** 2)
        Phi2 = math.atan2(ray_dir[2], ray_dir[0])
        Phi = Phi2 - Phi1
        Psi = cosTheta * cosThetaStar + sinTheta * sinThetaStar * math.cos(Phi)

        NPsi = (1.63 + 53.7 * math.exp(-5.49 * Psi) + 2.04 * math.cos(Psi) ** 2 * cosThetaStar) * (1 - math.exp(-1.90 * secTheta)) * (1 - math.exp(-0.53 * secThetaStar))
        sum_L_DIFF += h / len(V[0])
        sum_S_DIFF += h * NPsi

        F_PAR_DIFF = S_PAR_DIFF * NPsi
        F_NIR_DIFF = S_NIR_DIFF * NPsi
        F_L_DIFF   = S_L_DIFF / len(V[0])

        rays.append([ray_dir, dir_inv, F_PAR_DIFF, F_NIR_DIFF, F_L_DIFF])
    
    for i in range(len(rays)):
        rays[i][2] /= sum_S_DIFF
        rays[i][3] /= sum_S_DIFF
        rays[i][4] /= sum_L_DIFF

    return V, rays

def FindSunRays(sun_dir, tri, bounds, kdnodes, kdtriangles, root, depth, E_permanent, E_temporary):
    MAX_REFLECTIONS = 0
    print("Creating rays")
    sun_dir /= np.linalg.norm(sun_dir)
    dir_inv = np.array([divideHost(1.0, sun_dir[i]) for i in range(3)])
    
    rays = list()
    for t in tri:
        p = (3 * t['u'] + t['v'] + t['w']) / 3.0
        rays.append((p, sun_dir * (-1), dir_inv * (-1), [0.0, 0.0, 0.0]))
    nrays = len(rays)
    rays = np.array(rays, dtype = ray_dt)

    segments = np.array([[seg_sample] * (MAX_REFLECTIONS + 1)] * nrays, dtype = numba_seg_dt)
    stacks = np.array([[stack_sample] * depth] * nrays, dtype = numba_stack_dt)

    kdnodes_global_mem = cuda.to_device(kdnodes)
    kdtriangles_global_mem = cuda.to_device(kdtriangles)
    stacks_global_mem = cuda.to_device(stacks)

    threadsPerBlock = 4096
    numBlocks = math.ceil(1.0 * nrays / threadsPerBlock)

    print("Tracing rays")
    a = datetime.datetime.now()
    addRayKD2[threadsPerBlock, numBlocks](tri, root, rays, bounds, stacks_global_mem, kdnodes_global_mem, kdtriangles_global_mem, segments, MAX_REFLECTIONS, E_permanent, E_temporary)
    b = datetime.datetime.now()
    c = b - a
    print("Finished tracing in", c)

    view = list()
    for i in range(len(tri)):
        if segments[i][0]['which'] == -1:
            view.append(1.0)
        else:
            view.append(0.25)
        
    return view

def CreateSunRays(sun_dir, tri, view, path):
    sun_dir /= np.linalg.norm(sun_dir)
    dir_inv = np.array([divideHost(1.0, sun_dir[i]) for i in range(3)])

    rays = list()
    f = open(path, 'wb')
    for i in range(len(tri)):
        t = tri[i]
        if view[i] == 1.0:
            p = (3 * t['u'] + t['v'] + t['w']) / 3.0
            p += sun_dir * (50000.0 - p[1]) / sun_dir[1]
            factor = -(t['n'].dot(sun_dir))
            projA = t['A'] * factor

            if factor < 0:
                continue

            rays.append((p, sun_dir, dir_inv, [S_PAR_DIR * projA, #PAR, dir
                                               S_NIR_DIR * projA, #NIR, dir
                                               0.0]))             #L  , dir
            if len(rays) == BATCH_SIZE:
                np.save(f, np.array(rays, dtype = ray_dt))
                rays = list()

    if len(rays) != 0:
        np.save(f, np.array(rays, dtype = ray_dt))
    np.save(f, np.array([], dtype = ray_dt))

    f.close()

def inv(a):
    if a != 0.0:
        return 1.0 / a
    return INFINITY

def FileToRays(path, num_batches):
    rays = np.array([], dtype = ray_dt)
    with open(path, 'rb') as f:
        for i in range(num_batches):
            batch_rays = np.load(f)
            batch_rays = np.array(batch_rays, dtype = ray_dt)
            rays = np.concatenate((rays, batch_rays), axis = 0)
    return rays

def CreateEmittedRays(tri, raysPerTri, pathEmitted, pathPreDiffuse, sun_dir, VARIATIONS):
    V, Vrays = GenerateVectors(2 * raysPerTri + 1, sun_dir, VARIATIONS)
    rays  = list()
    rays2 = list()
    
    f = open(pathEmitted, 'wb')
    g = open(pathPreDiffuse, 'wb')
    a = datetime.datetime.now()

    for index in range(len(tri)):
        randomOffset = index % VARIATIONS
        t = tri[index]
        if math.isnan(t['n'][0]):
            continue
        n, x, y = t['n'], t['x'], t['y']

        p = (3 * t['u'] + t['v'] + t['w']) / 3.0
        for h, hprime, sinPhi, cosPhi in V[randomOffset]:
            ray_dir = h * n + hprime * (sinPhi * y + cosPhi * x)
            dir_inv = np.array([inv(ray_dir[i]) for i in range(3)])
            energy  = 2 * h / (raysPerTri + 1) * t['A']#L, eps
            
            rays.append((p, ray_dir, dir_inv, [energy, energy, energy], index))
            if len(rays) == BATCH_SIZE:
                np.save(f, np.array(rays, dtype = big_ray_dt))
                rays = list()
        if len(rays) + raysPerTri > BATCH_SIZE:
            np.save(f, np.array(rays, dtype = big_ray_dt))
            rays = list()

        for ray_dir, dir_inv, E_PAR_DIFF, E_NIR_DIFF, E_L_DIFF in Vrays:
            projA = ray_dir.dot(t['n']) * t['A']
            if projA > 0:
                rays2.append((p, ray_dir, dir_inv, [E_PAR_DIFF * projA,
                                                    E_NIR_DIFF * projA,
                                                    E_L_DIFF * projA], index))
                if len(rays2) == BATCH_SIZE:
                    np.save(g, np.array(rays2, dtype = big_ray_dt))
                    rays2 = list()

    if len(rays) != 0:
        np.save(f, np.array(rays, dtype = big_ray_dt))
    np.save(f, np.array([], dtype = big_ray_dt))
    if len(rays2) != 0:
        np.save(g, np.array(rays2, dtype = big_ray_dt))
    np.save(g, np.array([], dtype = big_ray_dt))

    b = datetime.datetime.now()
    c = b - a
    print(c, "seconds of extreme sadness")

    f.close()
    g.close()

def CreateEmittedPath(path, dstPath, frcPath, frcRPath, tri, bounds, kdnodes, kdtriangles, root, depth, E_permanent, E_temporary, raysPerTri):
    MAX_REFLECTIONS = 0
    f = open(path, 'rb')
    g = open(dstPath, 'wb')

    stacks = np.array([[stack_sample] * depth] * BATCH_SIZE, dtype = numba_stack_dt)
    batch_segments = np.array([[seg_sample] * (MAX_REFLECTIONS + 1)] * BATCH_SIZE, dtype = numba_seg_dt)

    kdnodes_global_mem = cuda.to_device(kdnodes)
    kdtriangles_global_mem = cuda.to_device(kdtriangles)
    stacks_global_mem = cuda.to_device(stacks)

    print("Tracing rays")
    a = datetime.datetime.now()
    nextInRg, i = 0, 0

    frac  = np.zeros((len(tri), ), dtype = np.float64)
    fracR = np.zeros((len(tri),3), dtype = np.float64)

    batch_rays = np.load(f)
    rays = list()
    while len(batch_rays):
        batch_rays = np.array(batch_rays, dtype = big_ray_dt)

        addRayKD1[THREADS_PER_BLOCK, NUM_BLOCKS](tri, root, batch_rays, bounds, stacks_global_mem, kdnodes_global_mem, kdtriangles_global_mem, batch_segments, MAX_REFLECTIONS, E_permanent, E_temporary)

        groups = len(batch_rays) // raysPerTri
        for group in range(groups):
            start = group * raysPerTri
            end = (group + 1) * raysPerTri
            for j in range(start, end):
                if batch_segments[j][0]['which'] != -1:
                    s = copy.deepcopy(batch_rays[j]['index'])
                    t = copy.deepcopy(batch_segments[j][0]['which'])
                    E = np.array(copy.deepcopy(batch_rays[j]['E']))

                    factor = -batch_rays[j]['r'].dot(tri[t]['n'])
                    if factor > 0:
                        E /= tri[t]['A']
                        frac[s] += 1
                        fracR[t] += E
                        rays.append((s, t, E))
                        if len(rays) == BATCH_SIZE:
                            np.save(g, np.array(rays, dtype = small_ray_dt))
                            rays = list()

        batch_rays = np.load(f)

    if len(rays) != 0:
        np.save(g, np.array(rays, dtype = small_ray_dt))
    np.save(g, np.array([], dtype = small_ray_dt))
    f.close()
    g.close()

    l = open(frcPath, 'wb')
    for i in range(len(tri)):
        frac[i] /= raysPerTri
    np.save(l, frac)
    l.close()

    l = open(frcRPath, 'wb')
    np.save(l, fracR)
    l.close()

    b = datetime.datetime.now()
    c = b - a
    print("Finished tracing in", c)

def CreateDiffuseRays(path, dstPath, dstPath2, tri, bounds, kdnodes, kdtriangles, root, depth, E_permanent, E_temporary):
    MAX_REFLECTIONS = 0
    f = open(path, 'rb')
    g = open(dstPath, 'wb')
    h = open(dstPath2, 'wb')

    stacks = np.array([[stack_sample] * depth] * BATCH_SIZE, dtype = numba_stack_dt)
    batch_segments = np.array([[seg_sample] * (MAX_REFLECTIONS + 1)] * BATCH_SIZE, dtype = numba_seg_dt)

    kdnodes_global_mem = cuda.to_device(kdnodes)
    kdtriangles_global_mem = cuda.to_device(kdtriangles)
    stacks_global_mem = cuda.to_device(stacks)

    print("Tracing rays")
    a = datetime.datetime.now()
    batch_rays = np.load(f)
    rays = list()
    rays2 = list()
    
    while len(batch_rays):
        batch_rays = np.array(batch_rays, dtype = big_ray_dt)
        addRayKD1[THREADS_PER_BLOCK, NUM_BLOCKS](tri, root, batch_rays, bounds, stacks_global_mem, kdnodes_global_mem, kdtriangles_global_mem, batch_segments, MAX_REFLECTIONS, E_permanent, E_temporary)

        for j in range(len(batch_rays)):
            if batch_segments[j][0]['which'] == -1:
                p = np.array(copy.deepcopy(batch_rays[j]['p']))
                r = np.array(copy.deepcopy(batch_rays[j]['r']))
                i = np.array(copy.deepcopy(batch_rays[j]['i']))
                E = np.array(copy.deepcopy(batch_rays[j]['E']))
                index = copy.deepcopy(batch_rays[j]['index'])

                p += r * 5000.0
                r = -r
                i = -i
                
                rays.append((p, r, i, E))
                rays2.append((index, E))

                if len(rays) == BATCH_SIZE:
                    np.save(g, np.array(rays, dtype = ray_dt))
                    np.save(h, np.array(rays2, dtype = tiny_ray_dt))
                    rays = list()
                    rays2 = list()

        batch_rays = np.load(f)

    if len(rays) != 0:
        np.save(g, np.array(rays, dtype = ray_dt))
        np.save(h, np.array(rays2, dtype = tiny_ray_dt))
    np.save(g, np.array([], dtype = ray_dt))
    np.save(h, np.array([], dtype = tiny_ray_dt))

    b = datetime.datetime.now()
    c = b - a
    print("Finished tracing in", c)

    f.close()
    g.close()
    h.close()
