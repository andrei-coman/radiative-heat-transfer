import numpy as np
from functools import cmp_to_key
import datetime
import copy
import math
from math import isnan
from numba import float64
from numba import float32
from numba import int32
import numba
from numba import none
from numba import cuda
from numba import njit

kdnode_dt    = np.dtype([('value', np.float64),
                         ('pointer1', np.int32),
                         ('pointer2', np.int32),
                         ('axis', np.int8),
                         ('isLeaf', np.bool_)])
numba_kdnode_dt = numba.from_dtype(kdnode_dt)

ray_dt       = np.dtype([('p', np.float64, (3,)),
                         ('r', np.float64, (3,)),
                         ('i', np.float64, (3,)),
                         ('E', np.float64, (3,))])
numba_ray_dt = numba.from_dtype(ray_dt)

tiny_ray_dt  = np.dtype([('t', np.int64),
                         ('E', np.float64, (3,))])
numba_tiny_ray_dt = numba.from_dtype(tiny_ray_dt)

small_ray_dt = np.dtype([('s', np.int32),
                         ('t', np.int32),
                         ('E', np.float64, (3,))])
numba_small_ray_dt = numba.from_dtype(small_ray_dt)

big_ray_dt   = np.dtype([('p', np.float64, (3,)),
                         ('r', np.float64, (3,)),
                         ('i', np.float64, (3,)),
                         ('E', np.float64, (3,)),
                         ('index', np.int64)])
numba_big_ray_dt = numba.from_dtype(big_ray_dt)

stack_dt     = np.dtype([('tmin', np.float64),
                         ('tmax', np.float64),
                         ('index', np.int32)])
numba_stack_dt = numba.from_dtype(stack_dt)
stack_sample = (0, 0.0, 0.0)

seg_dt       = np.dtype([('a', np.float32, (3,)),
                         ('b', np.float32, (3,)),
                         ('which', np.int32)])
numba_seg_dt = numba.from_dtype(seg_dt)
seg_sample = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0)

tri_dt       = np.dtype([('u', np.float64, (3,)),
                         ('v', np.float64, (3,)),
                         ('w', np.float64, (3,)),
                         ('n', np.float64, (3,)),
                         ('x', np.float32, (3,)),
                         ('y', np.float32, (3,)),
                         ('d', np.float64),
                         ('A', np.float32),
                         ('T', np.float32),
                         ('E_to_emit', np.float32, (3,)),
                         ('eps', np.float32, (3,)),
                         ('alpha', np.float64, (3,)),
                         ('k', np.float64)])
numba_tri_dt = numba.from_dtype(tri_dt)

INFINITY = 1000000000000000000.0
EPS = 0.001
EPS2 = 0.0001
LIMIT = 0

def BuildTree(trivec, path):
    print("Building KdTree")
    #trivec = list()
    kdnodes = list()
    kdtriangles = list()

    a = datetime.datetime.now()
    root, depth = BuildKdTree(trivec, list(range(len(trivec))), kdnodes, kdtriangles)
    b = datetime.datetime.now()
    c = b - a
    kdnodes = np.array(kdnodes, dtype = kdnode_dt)
    kdtriangles = np.array(kdtriangles, dtype = np.int32)
    
    with open(path, 'wb') as f:
        np.save(f, kdnodes)
        np.save(f, kdtriangles)
        np.save(f, np.array(Bounds(trivec)))
    print(root, depth, "Finished building KdTree in", c)

def LoadKdTree(path):
    with open(path, 'rb') as f:
        kdnodes = np.load(f)
        kdtriangles = np.load(f)
        bounds = np.load(f)

        root = len(kdnodes) - 1
        depth = dfs(root, kdnodes)

    kdnodes = np.array(kdnodes, dtype = kdnode_dt)
    kdtriangles = np.array(kdtriangles, dtype = np.int32)

    return (bounds, kdnodes, kdtriangles, root, depth)

#------------------------------------------------------------

@cuda.jit(device = True)
def divide(a, b):
    if b != 0.0:
        return  a / b
    elif a > 0.0:
        return +INFINITY
    else:
        return -INFINITY

def divideHost(a, b):
    if b != 0.0:
        return  a / b
    elif a > 0.0:
        return +INFINITY
    else:
        return -INFINITY

@cuda.jit(device = True)
def order(relCoord, r, left, right): #review for equal
    if relCoord > 0:
        return left, right
    if relCoord < 0:
        return right, left
    if r > 0:
        return left, right
    else:
        return right, left

@cuda.jit(device = True)
def intersectTR(T, ray):
    t = INFINITY

    u = T['u']
    v = T['v']
    w = T['w']
    n = T['n']
    d = T['d']
    p = ray['p']
    r = ray['r']

    ndotr = n[0] * r[0] + n[1] * r[1] + n[2] * r[2]
    ndotp = n[0] * p[0] + n[1] * p[1] + n[2] * p[2]
    dminusndotp = d - ndotp
    dprodr = ndotr * dminusndotp

    if ndotr == 0 or dprodr < 0:
        t = INFINITY
    else:
        t = dminusndotp / ndotr

        i1, i2 = 1, 2
        if v[0] * w[1] != w[0] * v[1]:
            i1, i2 = 0, 1
        elif v[0] * w[2] != w[0] * v[2]:
            i1, i2 = 0, 2
        xx = w[i1] * v[i2] - w[i2] * v[i1]

        p1 = p[i1] + t * r[i1] - u[i1]
        p2 = p[i2] + t * r[i2] - u[i2]
        yy = p1 * v[i2] - p2 * v[i1]
        tt = p1 * w[i2] - p2 * w[i1]

        status = 0
        
        if xx >= 0:
            status = (yy >= 0 and tt <= 0 and yy - tt <= xx)
        else:
            status = (yy < 0 and tt > 0 and yy - tt >= xx)

        if not status:
            t = INFINITY
        #t = status * t + (1 - status) * INFINITY
    
    return t

#---------------------------------------------------------------
#http://www.irisa.fr/prive/kadi/Sujets_CTR/kadi/Kadi_sujet2_article_Kdtree.pdf

class Plane:
    def __init__(self, _axis, _x):
        self.axis = _axis
        self.x = _x

class Event:
    def __init__(self, _x, _tau):
        self.x = _x
        self.tau = _tau

def compareE(a, b):
    if a.x < b.x:
        return -1
    if a.x > b.x:
        return +1
    if a.tau <= b.tau:
        return -1
    return 1

#---------------------------------------------------------------

def Bounds(T):
    minx, maxx = +INFINITY, -INFINITY
    miny, maxy = +INFINITY, -INFINITY
    minz, maxz = +INFINITY, -INFINITY
    points = []
    for t in T:
        points.append(t['u'])
        points.append(t['v'])
        points.append(t['w'])
    for p in points:
        minx, maxx = min(minx, p[0]), max(maxx, p[0])
        miny, maxy = min(miny, p[1]), max(maxy, p[1])
        minz, maxz = min(minz, p[2]), max(maxz, p[2])
    return [[minx, maxx], [miny, maxy], [minz, maxz]]

def IntersectBounds(B1, B2):
    minx = max(B1[0][0], B2[0][0])
    maxx = min(B1[0][1], B2[0][1])
    miny = max(B1[1][0], B2[1][0])
    maxy = min(B1[1][1], B2[1][1])
    minz = max(B1[2][0], B2[2][0])
    maxz = min(B1[2][1], B2[2][1])

    return [[minx, maxx], [miny, maxy], [minz, maxz]]

def DumbClipTriangleToBox(t, V):
    return IntersectBounds(V, Bounds([t]))

KT = 1
KI = 128
def C(PL, PR, NL, NR):
    return KT + KI * (PL * NL + PR * NR)

def SplitBox(V, p):
    VL = copy.deepcopy(V)
    VL[p.axis][1] = p.x
    VR = copy.deepcopy(V)
    VR[p.axis][0] = p.x
    return (VL, VR)

def SA(V):
    x = V[0][1] - V[0][0]
    y = V[1][1] - V[1][0]
    z = V[2][1] - V[2][0]
    return 2 * (x * y + y * z + z * x)

def SAH(p, V, NL, NR, NP):
    VL, VR = SplitBox(V, p) #split the box V by plane p
    PL = divideHost(SA(VL), SA(V)) #conditional probability to hit VL
    PR = divideHost(SA(VR), SA(V)) #conditional probability to hit VR
    cpl = C(PL, PR, NL+NP, NR) #cost to put splitting plane on the left
    cpr = C(PL, PR, NL, NP+NR) #cost to put splitting plane on the right
    if cpl < cpr:
        return (cpl, 0) #0 -> left
    else:
        return (cpr, 1) #1 -> right

def TriangleSetPlane(T, p, side):
    S = []
    axis = p.axis
    x = p.x
    for t in T:
        u = t['u']
        v = t['v']
        w = t['w']
        
        su = (u[axis] - p.x) * side
        sv = (v[axis] - p.x) * side
        sw = (w[axis] - p.x) * side

        if su >= 0 and sv >= 0 and sw >= 0:
            S.append((u, v, w, t['n'], t['d']))

        elif su >= 0 and sv < 0 and sw < 0:
            vprime = u + (v - u) * (x - u[axis]) / (v[axis] - u[axis])
            wprime = u + (w - u) * (x - u[axis]) / (w[axis] - u[axis])
            S.append((u, vprime, wprime, t['n'], t['d']))
        elif sv >= 0 and sw < 0 and su < 0:
            wprime = v + (w - v) * (x - v[axis]) / (w[axis] - v[axis])
            uprime = v + (u - v) * (x - v[axis]) / (u[axis] - v[axis])
            S.append((v, wprime, uprime, t['n'], t['d']))
        elif sw >= 0 and su < 0 and sv < 0:
            uprime = w + (u - w) * (x - w[axis]) / (u[axis] - w[axis])
            vprime = w + (v - w) * (x - w[axis]) / (v[axis] - w[axis])
            S.append((w, uprime, vprime, t['n'], t['d']))

        elif su < 0 and sv >= 0 and sw >= 0:
            vprime = u + (v - u) * (x - u[axis]) / (v[axis] - u[axis])
            wprime = u + (w - u) * (x - u[axis]) / (w[axis] - u[axis])
            S.append((vprime, v, w, t['n'], t['d']))
            S.append((w, wprime, vprime, t['n'], t['d']))
        elif sv < 0 and sw >= 0 and su >= 0:
            wprime = v + (w - v) * (x - v[axis]) / (w[axis] - v[axis])
            uprime = v + (u - v) * (x - v[axis]) / (u[axis] - v[axis])
            S.append((wprime, w, u, t['n'], t['d']))
            S.append((u, uprime, wprime, t['n'], t['d']))
        elif sw < 0 and su >= 0 and sv >= 0:
            uprime = w + (u - w) * (x - w[axis]) / (u[axis] - w[axis])
            vprime = w + (v - w) * (x - w[axis]) / (v[axis] - w[axis])
            S.append((uprime, u, v, t['n'], t['d']))
            S.append((v, vprime, uprime, t['n'], t['d']))
    S = np.array(S, dtype = dt)
    return S

def ClipTriangleToBox(t, V):
    tri = [t]
    for axis in range(3):
        tri = TriangleSetPlane(tri, Plane(axis, V[axis][0]), +1)
        tri = TriangleSetPlane(tri, Plane(axis, V[axis][1]), -1)
    return Bounds(tri)

def FindPlane(V, bounds):
    Chat, phat, psidehat = +INFINITY, Plane(0, 0.0), 0
    
    for k in range(3): #for each dimension in turn
        E = [] #eventlist
        for B in bounds:
            if B[k][0] == B[k][1]:
                E.append(Event(B[k][0], 1)) #|
            else:
                E.append(Event(B[k][0], 2)) #+
                E.append(Event(B[k][1], 0)) #-

        E.sort(key = cmp_to_key(compareE))

        NL, NP, NR = 0, 0, len(B)
        i = 0
        while i < len(E):
            p = Plane(k, E[i].x)
            pplus, pminus, pbar = 0, 0, 0
            while i < len(E) and E[i].x == p.x and E[i].tau == 0:
                pminus += 1
                i += 1
            while i < len(E) and E[i].x == p.x and E[i].tau == 1:
                pbar += 1
                i += 1
            while i < len(E) and E[i].x == p.x and E[i].tau == 2:
                pplus += 1
                i += 1

            #move plane ONTO p 
            NP = pbar
            NR -= pbar
            NR -= pminus

            C, pside = SAH(p, V, NL, NR, NP)
            if C < Chat:
                Chat, phat, psidehat = C, p, pside

            #move plane OVER p
            NL += pplus
            NL += pbar
            NP = 0

    return (Chat, phat, psidehat)

def RecBuild(tri, T, V, kdnodes, kdtriangles, depth):
    if len(T) <= LIMIT:
        kdnodes.append((0.0, len(kdtriangles), 0, 0, True))
        kdtriangles.extend(T)
        kdtriangles.append(-1)
        return len(kdnodes) - 1, depth
    
    bounds = []
    for i in T:
        t = tri[i]
        B = DumbClipTriangleToBox(t, V)
        bounds.append(B)

    C, p, pside = FindPlane(V, bounds)
    VL, VR = SplitBox(V, p)
    
    if depth <= 7:
        print(VL)
        print(VR)
    TL, TR = [], []
    
    for i in range(len(T)):
        t = T[i]
        B = bounds[i]

        if B[p.axis][0] == B[p.axis][1] and B[p.axis][0] == p.x:
            if pside == 0:
                TL.append(t)
            else:
                TR.append(t)
        else:
            if B[p.axis][0] < p.x:
                TL.append(t)
            if B[p.axis][1] > p.x:
                TR.append(t)
    
    if p.x == V[p.axis][0] and len(TL) == 0 or p.x == V[p.axis][1] and len(TR) == 0:
        kdnodes.append((0.0, len(kdtriangles), 0, 0, True))
        kdtriangles.extend(T)
        kdtriangles.append(-1)
        return len(kdnodes) - 1, depth


    left, dleft   = RecBuild(tri, TL, VL, kdnodes, kdtriangles, depth + 1)
    right, dright = RecBuild(tri, TR, VR, kdnodes, kdtriangles, depth + 1)
    kdnodes.append((p.x, left, right, p.axis, False))
    
    return len(kdnodes) - 1, max(dleft, dright)

def BuildKdTree(tri, T, kdnodes, kdtriangles):
    V = Bounds(tri)
    return RecBuild(tri, T, V, kdnodes, kdtriangles, 0)

def dfs(idx, kdnodes):
    if kdnodes[idx]["isLeaf"]:
        return 0
    return 1 + max(dfs(kdnodes[idx]['pointer1'], kdnodes), 
                   dfs(kdnodes[idx]['pointer2'], kdnodes))

#---------------------------------------------------------------

@cuda.jit(device = True)
def intersect(bounds, ray):
    xtmin = (bounds[0][0] - ray['p'][0]) * ray['i'][0]
    xtmax = (bounds[0][1] - ray['p'][0]) * ray['i'][0]
    ytmin = (bounds[1][0] - ray['p'][1]) * ray['i'][1]
    ytmax = (bounds[1][1] - ray['p'][1]) * ray['i'][1]
    ztmin = (bounds[2][0] - ray['p'][2]) * ray['i'][2]
    ztmax = (bounds[2][1] - ray['p'][2]) * ray['i'][2]
    
    if xtmax < xtmin:
        xtmin, xtmax = xtmax, xtmin
    if ytmax < ytmin:
        ytmin, ytmax = ytmax, ytmin
    if ztmax < ztmin:
        ztmin, ztmax = ztmax, ztmin
    
    xtmax = max(xtmax, 0.0)
    xtmin = max(xtmin, 0.0)
    ytmax = max(ytmax, 0.0)
    ytmin = max(ytmin, 0.0)
    ztmax = max(ztmax, 0.0)
    ztmin = max(ztmin, 0.0)
    
    return (max(xtmin, ytmin, ztmin), min(xtmax, ytmax, ztmax))

#@cuda.jit((numba_tri_dt[:], int32, int32, float64[:,:], numba_ray_dt, numba_kdnode_dt[:], int32[:], numba_stack_dt[:,:]), device = True)
@cuda.jit(device = True)
def kdSearch(tri, numRay, splitIdx, bounds, ray, kdnodes, kdtriangles, stacks):
    top = 0
    tmin, tmax = intersect(bounds, ray)
    
    mint, which = INFINITY, -1
    foundHit = False
    bestT, bestTri = INFINITY, -1
    
    while not foundHit and splitIdx != -1:
        while not kdnodes[splitIdx]["isLeaf"]:
            split = kdnodes[splitIdx]
            axis = split['axis']
            thit = (split['value'] - ray['p'][axis]) * ray['i'][axis]
            
            left  = split['pointer1']
            right = split['pointer2']
            first, second = order(split['value'] - ray['p'][axis], ray['i'][axis], left, right)

            if tmax + EPS < thit or thit < 0:
                splitIdx = first
            elif thit < tmin - EPS:
                splitIdx = second
            else:
                stacks[numRay][top]['index'] = second
                stacks[numRay][top]['tmin'] = thit
                stacks[numRay][top]['tmax'] = tmax
                top += 1
                
                splitIdx = first
                tmax = thit
    
        leafIdx = splitIdx
        i = kdnodes[leafIdx]['pointer1']
        while kdtriangles[i] != -1:
            t = intersectTR(tri[kdtriangles[i]], ray)

            if EPS2 < t < bestT:
                bestT = t
                bestTri = kdtriangles[i]
            i += 1
        
        if bestT <= tmax + EPS:
            foundHit = True
            mint, which = bestT, bestTri
        elif top == 0:
            splitIdx = -1
        else:
            top -= 1
            splitIdx = stacks[numRay][top]['index']
            tmin = stacks[numRay][top]['tmin']
            tmax = stacks[numRay][top]['tmax']

    return mint, which

@cuda.jit(device = True)
def reflection(ray, t, T):
    tmp = 2 * (ray['r'][0] * T['n'][0] + ray['r'][1] * T['n'][1] + ray['r'][2] * T['n'][2])
    for i in range(3):
        ray['p'][i] = ray['p'][i] + t * ray['r'][i]
        ray['r'][i] = ray['r'][i] - tmp * T['n'][i]
        ray['i'][i] = divide(1.0, ray['r'][i])


