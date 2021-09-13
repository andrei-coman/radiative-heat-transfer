import ctypes
import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
import pygame
from pygame.locals import *
import numpy as np
import random

VERTEX_SHADER = """
#version 330
    uniform mat4 modelView;
    in vec4 position;
    in vec3 clr;

    varying vec3 vColor;
    void main() {
        gl_Position = modelView * position;
         vColor = clr;
    }
"""
FRAGMENT_SHADER = """
#version 330
    varying vec3 vColor;
    void main(){       
        gl_FragColor = vec4(vColor.x, vColor.y, vColor.z, 1.0f);
    }

"""
shaderProgram = None

def CreateBuffer(vertices):
    bufferdata = (ctypes.c_float*len(vertices))(*vertices) # float buffer
    buffersize = len(vertices)*4                           # buffer size in bytes

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, buffersize, bufferdata, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return vbo

def DrawBuffer(vbo, noOfVertices):
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    #binds vbo to GL_ARRAY_BUFFER
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 24, ctypes.c_void_p(0))
    glColorPointer(3, GL_FLOAT, 24, ctypes.c_void_p(12))
    #3 coordinates per vertex, of type float, 24 offset between vertices, 0 offset at start
    #3 coordinates per colour, of type float, 24 offset between colours, 12 offset at start 

    glDrawArrays(GL_POINTS, 0, noOfVertices)
    #draw GL_POINTS primitives, starting from 0 in the enabled array, with noOfVertices indices to be rendered

    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    #unbinds everything from GL_ARRAY_BUFFER

def initializeRenderer():
    sys.setrecursionlimit(10000)
    pygame.init()
    display = (3840, 2160)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    glLineWidth(3.0)
    #glPointSize(5.0)

    gluPerspective(45, (display[0]/display[1]), 100, 10000000000.0)
    glTranslatef(-50000.0, -50000.0, -200000.0)
    glRotatef(20, 1, 0, 0)

    global VERTEX_SHADER
    global FRAGMENT_SHADER
    global shaderProgram
    vertexshader = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragmentshader = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    shaderProgram = shaders.compileProgram(vertexshader, fragmentshader)
    glEnable(GL_DEPTH_TEST)
    glDepthMask(GL_TRUE)
    glDepthFunc(GL_LEQUAL)
    glDepthRange(0.0, 1.0)

def render(bufferTri, triangles, bufferLvs, leaves):
    global shaderProgram
    global VAO

    vbo = bufferTri
    sz = len(triangles) // 6

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glClearColor(1.0, 1.0, 1.0, 0.0)
    glClearDepth(1.0)

    #use shader
    glUseProgram(shaderProgram)

    #bind vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 24, ctypes.c_void_p(0))
    glColorPointer(3, GL_FLOAT, 24, ctypes.c_void_p(12))

    #input vertex position in vertex shader
    position = glGetAttribLocation(shaderProgram, 'position')
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, None)
    glEnableVertexAttribArray(position)

    color = glGetAttribLocation(shaderProgram, 'clr')
    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)

    #change uniform modelView matrix in vertex shader
    modelViewMatrix = (GLfloat * 16)()
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix)
    modelViewMatrix = np.array(list(modelViewMatrix), np.float32)
    loc = glGetUniformLocation(shaderProgram, 'modelView')
    glUniformMatrix4fv(loc, 1, GL_FALSE, modelViewMatrix)

    #draw triangles
    glDrawArrays(GL_TRIANGLES, 0, sz)

    #disable everything
    glDisableVertexAttribArray(position)
    glDisableVertexAttribArray(color)
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glUseProgram(0)

    #DrawBuffer(bufferLvs, len(leaves) // 6)

def addNoise(n, Tmin, Tmax, T):
    return n + np.array([random.uniform(0, 0.1), random.uniform(0, 0.1), random.uniform(0, 0.1)])

def normalColoring(n, Tmin, Tmax, T):
    return n

def darken(n, Tmin, Tmax, T):
    return 0.75 * n

def randomColoring(n, Tmin, Tmax, T):
    return np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])

def groundColoring(n, Tmin, Tmax, T):
    return np.array([195/256, 155/256, 119/256])

def tempColoring(n, Tmin, Tmax, T):
    Tmid = (Tmin + Tmax) / 2
    if T < Tmin:
        return np.array([0.0, 0.0, 1.0])
    elif T > Tmax:
        return np.array([1.0, 0.0, 0.0])
    elif T < Tmid:
        return np.array([0.0, (T - Tmin) / (Tmid - Tmin), (Tmid - T) / (Tmid - Tmin)])
    else:
        return np.array([(T - Tmid) / (Tmax - Tmid), (Tmax - T) / (Tmax - Tmid), 0.0])

def plotSegment(p, q, R, G, B):
    glColor3f(R, G, B)
    glBegin(GL_LINES)
    glVertex3f(p[0], p[1], p[2])
    glVertex3f(q[0], q[1], q[2])
    glEnd()

