import pylas

import rdp
import alphashape
import math
import numpy as np
import numpy.linalg as nla
import random
from scipy.spatial import Delaunay
from panda3d.core import Triangulator
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

import ctypes
import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
import pygame
from pygame.locals import *

zallow = 3       #min z threshold (to ignore ground artifacts)
deltaX = 2048    #discretization size along x
deltaY = 2048    #discretization size along y
deltaZ = 1024    #discretization size along z
threshold = 400  #building surface point count threshold
threshold2 = 150 #threshold for recursive calls
alphainv = 200.0 #radius for alphashapes
scale = 100.0    #scale for alphashapes (to avoid splitting of buildings)

class BuildingSegmenter:
    def __init__(self):
        self.triangles = list()
        self.ground = list()
        self.leaves = list()
        self.las = None
        self.GT = Triangulator()
        self.GPoly = list()
        self.GP = list()

    #------------------------------------------------------------------------------------------
    #private methods

    '''
    Create the concave hull for the top of a building
    @param {list} vSet - the set of points constituting the top of the building
    @returns {list} hulls - the set of points constituting the concave hull, in ccw order
    '''
    def __createHull(self, vSet):
        print("Creating building concave hull")
        print("Finding alpha")
        alpha = 1.0 / (alphainv / scale) #alpha parameter for alphashapes (alpha = 1/radius)

        print("Finding concave hull")
        #vSet = random.sample(vSet, min(len(vSet), 10000))
        points = [(point[0] / scale, point[1] / scale) for point in vSet]
        hulls = alphashape.alphashape(points, alpha)
        return hulls

    '''
    Add a triangle to the array of triangles
    @param {list} A - first point from the triangle
    @param {list} B - second point from the triangle
    @param {list} C - third point from the triangle
    @param {boolean} backfaces - add the backface?
    '''
    def __addTriangle(self, A, B, C, arr, backfaces = 0):
        AC = np.array(A) - np.array(C)          #CA vector
        BC = np.array(B) - np.array(C)          #CB vector
        n = np.cross(AC, BC)                    #cross product of sides
        n = n / nla.norm(n)                     #unit normal
        clr = list((n + [1.0, 1.0, 1.0]) / 2.0) #color triangle by normal

        arr.extend(A + clr)
        arr.extend(B + clr)
        arr.extend(C + clr)
        
        if backfaces:
            arr.extend(C + clr)
            arr.extend(B + clr)
            arr.extend(A + clr)

    '''
    Process the concave hull of the top of a building (triangulate and add to triangle array)
    @param {list} hull - the concave hull in ccw order
    @param {number} avgZ - the z-coord of the points; the hull is leveled
    @param {list} vSet - array of vertices; any vertices to be displayed are added here
    @returns {list} vSet - the updated array of vertices
    '''
    def __processHull(self, hull, avgZ):
        x, y  = hull.exterior.coords.xy #scale back the coordinates of the hull
        
        pts = [[x[i], y[i]] for i in range(len(x))]
        pts = rdp.rdp(pts, epsilon = 2.0)
        x = [pts[i][0] for i in range(len(pts))]
        y = [pts[i][1] for i in range(len(pts))]
        
        if len(pts) < 3:
            return

        x = [scale * coord for coord in x]
        y = [scale * coord for coord in y]

        coords = [(x[i], y[i]) for i in range(len(pts))]
        polygon = Polygon(coords)
        self.GPoly.append(polygon)

        T = Triangulator() #triangulator
        #self.GT.begin_hole()
        for i in range(len(x)):
            p = T.addVertex(x[i], y[i]) #add vertices to the triangulator
            T.addPolygonVertex(p) #specify polygon order of vertices
            
            '''
            self.GP.append([x[i], y[i]])
            p = self.GT.addVertex(x[i], y[i])
            self.GT.addHoleVertex(p)
            '''
        T.triangulate() #triangulate the vertices

        for i in range(len(x)): #add sides of the building
            j = i + 1
            if j == len(x):
                j = 0

            self.__addTriangle([x[j], avgZ, y[j]],
                             [x[i], avgZ, y[i]],
                             [x[i], 0,    y[i]],
                             self.triangles)
            self.__addTriangle([x[i], 0,    y[i]],
                             [x[j], 0,    y[j]],
                             [x[j], avgZ, y[j]],
                             self.triangles)
            
        tri = T.getNumTriangles()
        for i in range(tri): #add top of the building
            v0 = T.getTriangleV0(i)
            v1 = T.getTriangleV1(i)
            v2 = T.getTriangleV2(i)
            
            self.__addTriangle([x[v2], avgZ, y[v2]],
                               [x[v1], avgZ, y[v1]],
                               [x[v0], avgZ, y[v0]],
                               self.triangles)
    
    def __processHole(self, hole):
        x, y  = hole.exterior.coords.xy
        self.GT.begin_hole()
        for i in range(len(x)):
            self.GP.append([x[i], y[i]])
            p = self.GT.addVertex(x[i], y[i])
            self.GT.addHoleVertex(p)
            

    '''
    Fill (DFS-ish) method to group together boxes that correspond to the same building top
    The function is called on a box for which the number of points exceeds threshold
    The function then recursively calls all the neighbours; for each such neighbour, the points are added
    If the neighbour does not exceed threshold2, no further calls are made
    @param {dict} boxes - dictionary of boxes
    @param {number} key - key of the current box in the dictionary
    @param {list} vSet - the vertices that are part of the current building top
    '''
    def __fill(self, boxes, key, vSet):
        for point in boxes[key]:
            vSet.append((point[0], point[1], point[2]))
        length = len(boxes[key])
        del boxes[key]
        if length < threshold2:
            return None

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    nkey = (key[0] + dx, key[1] + dy, key[2] + dz)
                    if nkey in boxes:
                        self.__fill(boxes, nkey, vSet)

    '''
    Discretizes a point
    @param {list} point - point to be discretized
    @returns {tuple} - discretized coordinates
    '''
    def __discretize(self, point):
        return ((point[0] // deltaX),
                (point[1] // deltaY),
                (point[2] // deltaZ))

    '''
    Discretizes the loaded points
    @returns {dict} - dictionary of discrete boxes
    '''
    def __DiscretizePoints(self): #discretizes the points loaded by LoadPoints
        boxes = {}
        print("Discretizing z coords")
        for point in self.las.points:
            dp = self.__discretize(point)
            z = dp[2] - self.offset #the first box starts at z = 0
            if z > zallow:
                if dp not in boxes:
                    boxes[dp] = list()
                boxes[dp].append(point)
        return boxes

    #------------------------------------------------------------------------------------------
    #public methods
    
    '''
    Loads the points from the "path" laz file
    '''
    def LoadPoints(self, path):
        fh = pylas.open(path)
        vlr = fh.read_vlrs()
        self.las = fh.read()

        self.xmax = -1000000000000
        self.xmin = +1000000000000
        self.ymax = -1000000000000
        self.ymin = +1000000000000
        
        for point in self.las.points:
            if point[0] < self.xmin:
                self.xmin = point[0]
            if point[0] > self.xmax:
                self.xmax = point[0]
            if point[1] < self.ymin:
                self.ymin = point[1]
            if point[1] > self.ymax:
                self.ymax = point[1]
        
        self.zmax = self.las.header.z_max * 100
        self.zmin = self.las.header.z_min * 100
        self.zboxmax = int(self.zmax // deltaZ)
        self.zboxmin = int(self.zmin // deltaZ)
        self.offset = self.zboxmin

    def CreateLeaves(self):
        clr = [0.0, 1.0, 0.0]
        for point in self.las.points:
            clr = [0.0, random.uniform(0, 1), 0.0]
            self.leaves.extend([point[0], point[2], point[1]] + clr)
        return self.leaves

    '''
    Creates the building surfaces from the loaded points
    Every triangle is added to the list triangles
    '''
    def CreateSurfaces(self):
        boxes = self.__DiscretizePoints()

        print("Processing boxes")
        keys = tuple(boxes.keys())
        count = 0
        for box_key in keys:
            if box_key in boxes and len(boxes[box_key]) >= threshold:
                count += 1
                
                print("Detected building", count)
                print("Expanding buiding", count)
                vSet = list()
                self.__fill(boxes, box_key, vSet)

                #if count > 1:
                #    continue

                avgZ = sum([point[2] for point in vSet]) // len(vSet) + 300.0 #to be changed
                hulls = self.__createHull(vSet)
        
                if hulls.type == "Polygon":
                    hulls = [hulls]
                for hull in hulls:
                    self.__processHull(hull, avgZ)

    def CreateGround(self):
        u = cascaded_union(self.GPoly)
        for hole in u:
            self.__processHole(hole)

        self.GP.append([self.xmin, self.ymin])
        self.GP.append([self.xmax, self.ymin])
        self.GP.append([self.xmax, self.ymax])
        self.GP.append([self.xmin, self.ymax])

        p = self.GT.addVertex(self.xmin, self.ymin)
        self.GT.addPolygonVertex(p)
        p = self.GT.addVertex(self.xmax, self.ymin)
        self.GT.addPolygonVertex(p)
        p = self.GT.addVertex(self.xmax, self.ymax)
        self.GT.addPolygonVertex(p)
        p = self.GT.addVertex(self.xmin, self.ymax)
        self.GT.addPolygonVertex(p)

        self.GT.triangulate()
        tri = self.GT.getNumTriangles()
        for i in range(tri):
            v0 = self.GT.getTriangleV0(i)
            v1 = self.GT.getTriangleV1(i)
            v2 = self.GT.getTriangleV2(i)
            
            self.__addTriangle([self.GP[v2][0], 0, self.GP[v2][1]],
                               [self.GP[v1][0], 0, self.GP[v1][1]],
                               [self.GP[v0][0], 0, self.GP[v0][1]],
                               self.ground)

    '''
    Returns the triangles list
    returns {list} triangles - list of created triangles
    '''
    def getTriangles(self):
        return self.triangles
    
    '''
    Returns the leaves list
    returns {list} leaves - list of created leaves
    '''
    def getLeaves(self):
        return self.leaves

    '''
    Returns the ground-triangles list
    returns {list} ground - list of ground-triangles
    '''
    def getGround(self):
        return self.ground
