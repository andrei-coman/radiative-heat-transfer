import numpy as np
import struct
from vpython import *
from BuildingSegmenter import *

import queue
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

class Triangle(object):
    def __init__(self, _u, _v, _w, _n):
        self.vertices = np.array([_u, _v, _w])
        self.n = _n
    def normal_color(self):
        return (self.n + np.array([1, 1, 1])) / 2

def BytestringToVec3(bytestring):
    u = struct.unpack('<f', bytestring[0: 4])
    v = struct.unpack('<f', bytestring[4: 8])
    w = struct.unpack('<f', bytestring[8:12])
    return np.array([u[0], v[0], w[0]])

def ReadSTL(fileName):
    #fileName = fileName.lower()
    if fileName[-4:] != ".stl":
        raise Exception("Invalid file format.")
    try:
        fin = open(fileName, "rb")
    except:
        raise Exception("Failed to read STL file.")
    
    header = fin.read(80) #ignore
    number_faces = struct.unpack('<i', fin.read(4))[0]
    faces = []

    for i in range(number_faces):
        triangle_data = fin.read(50)
        normal  = BytestringToVec3(triangle_data[0:12])
        vertex1 = BytestringToVec3(triangle_data[12:24])
        vertex2 = BytestringToVec3(triangle_data[24:36])
        vertex3 = BytestringToVec3(triangle_data[36:48])
        faces.append(Triangle(vertex1, vertex2, vertex3, normal))
        
    return faces

def toArray(faces, tri, offset, colorFunc):
    triangles = list()
    for i in range(len(faces)):
        face = faces[i]
        u = list(face.vertices[0])
        v = list(face.vertices[1])
        w = list(face.vertices[2])
        n = list((face.n + np.array([1, 1, 1])) / 2.0)
        n = list(colorFunc(np.array(n), tri[offset + i]['T'], offset + i))
        triangles += u + n
        triangles += v + n
        triangles += w + n
    return triangles

def distSquared(u, v):
    return (u[0] - v[0])**2 + (u[1] - v[1])**2 + (u[2] - v[2])**2

def fragmentTriangle(u, v, w, maxLen):
    fragments = list()
    q = queue.Queue()
    q.put([u, v, w])
    while not q.empty():
        tri = q.get()
        u, v, w = tuple(tri)
        uv = distSquared(u, v)
        vw = distSquared(v, w)
        wu = distSquared(w, u)

        order = [u, v, w]
        if uv >= vw and uv >= wu:
            order = [u, v, w]
        if vw >= wu and vw >= uv:
            order = [v, w, u]
        if wu >= uv and wu >= vw:
            order = [w, u, v]

        maxSide = max(uv, vw, wu)
        
        if maxSide <= maxLen**2:
            fragments.append([u, v, w])
        else:
            u, v, w = tuple(order)
            x = list((np.array(u) + np.array(v)) / 2.0)
            q.put([u, x, w])
            q.put([x, v, w])
        
        #fragments.append([u, v, w])

    return fragments

def fragmentTriangles(triangles, maxLen):
    newTriangles = list()
    num_faces = len(triangles) // 18

    for i in range(num_faces):
        u = triangles[18*i     :18*i +  3]
        v = triangles[18*i +  6:18*i +  9]
        w = triangles[18*i + 12:18*i + 15]

        uw = np.array(u) - np.array(w)
        vw = np.array(v) - np.array(w)
        n = np.cross(uw, vw)
        n = list(n / nla.norm(n))

        fragments = fragmentTriangle(u, v, w, maxLen)
        for fragment in fragments:
            u = fragment[0]
            v = fragment[1]
            w = fragment[2]
            newTriangles.append(n + u + v + w)
    return newTriangles

def writeTrianglesToFile(path, triangles):
    fh = open(path, "wb")
    fh.write(bytearray([0] * 80)) #header
    num_faces = len(triangles)
    fh.write(struct.pack('@i', num_faces))

    for triangle in triangles:
        for coord in triangle:
            fh.write(struct.pack('<f', coord))
        fh.write(struct.pack('<h', 0))
    fh.close()

def lazToStl(path, buildingsPath, groundPath, maxLenBuildings, maxLenGround):
    BS = BuildingSegmenter()
    BS.LoadPoints(path)
    BS.CreateSurfaces()
    triangles = BS.getTriangles()
    triangles = fragmentTriangles(triangles, maxLenBuildings)
    
    BS.CreateGround()
    ground = BS.getGround()
    ground = fragmentTriangles(ground, maxLenGround)

    writeTrianglesToFile(buildingsPath, triangles)
    writeTrianglesToFile(groundPath, ground)

def plot3d(faces):
    minx = min([face.vertices[i][0] for face in faces for i in range(3)])
    miny = min([face.vertices[i][1] for face in faces for i in range(3)])
    minz = min([face.vertices[i][2] for face in faces for i in range(3)])
    maxx = max([face.vertices[i][0] for face in faces for i in range(3)])
    maxy = max([face.vertices[i][1] for face in faces for i in range(3)])
    maxz = max([face.vertices[i][2] for face in faces for i in range(3)])
    
    ax = a3.Axes3D(pl.figure())
    ax.set_xlim3d(minx, maxx)
    ax.set_ylim3d(miny, maxy)
    ax.set_zlim3d(minz, maxz)
    i = 0
    for face in faces:
        i += 1
        tri = a3.art3d.Poly3DCollection(face.vertices)
        tri.set_color(face.normal_color())
        #tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    pl.show()
    return

