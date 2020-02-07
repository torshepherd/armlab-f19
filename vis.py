import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
import kinematics

R = [1, 0, 0]
G = [0, 1, 0]
B = [0, 0, 1]
BLACK = [0, 0, 0]

COLORS = [R, G, B, BLACK]
SCALAR0 = 50


class Origin:

    def __init__(self, o, x0, y0):
        '''
        To define an origin, pass the vector (as numpy arrays) containing the offset from the standard origin and the direction of x (in 0) and y (in zero)
        '''
        self.o = o
        self.x0 = x0
        self.y0 = y0
        self.z0 = np.cross(x0, y0)


def draw_axes(o, x0, y0, size=1, color=[]):
    z0 = np.cross(x0, y0)
    vertices = [list(o), list(o + size * x0),
                list(o + size * y0), list(o + size * z0)]
    edges = [[0, 1], [0, 2], [0, 3]]
    glLineWidth(2)
    glBegin(GL_LINES)
    for edge_i in range(len(edges)):
        if color == []:
            glColor3fv(COLORS[edge_i])
        else:
            glColor3fv(color)
        edge = edges[edge_i]
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()


def draw_stickbot(list_of_origins):
    glLineWidth(1)
    glBegin(GL_LINES)
    glColor3fv([0, 0, 0])
    edges = [[0, 1], [1, 2], [2, 3], [3, 4]]
    for edge in edges:
        for vertex in edge:
            glVertex3fv(list_of_origins[vertex])
    glEnd()


def T2o(T):
    return T[0:3, 3] / SCALAR0


def o2T(R, p):
    return np.vstack([np.hstack([R, p]), np.array([0, 0, 0, 1])])


def draw_T(T, color=[]):
    draw_axes(T2o(T), T[0:3, 0], T[0:3, 1], color=color)


def draw_robot(joints):
    Tlist = []
    for i in range(len(joints)+1):
        Tlist.append(kinematics.FK_dh(np.array(joints), i).matrix)
    olist = []
    for T in Tlist:
        draw_T(T)
        olist.append(list(T2o(T)))
    draw_stickbot(olist)


def display_robot(joints, animation_name='', T=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])):
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glClearColor(1, 1, 1, 1)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, -2.0, -20)
    glRotatef(-70, 1, 0, 0)
    glRotatef(180, 0, 0, 1)
    # glRotatef(-60, 1, 0, 0)
    # glRotatef(-30, 0, 0, 1)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE: 
                    pygame.quit()
                    pygame_quit = True
            elif event.type == pygame.QUIT:
                pygame.quit()
                pygame_quit = True

        glRotatef(0.5, 0, 0, 1)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_axes(np.array([0, 0, 0]), np.array(
            [1, 0, 0]), np.array([0, 1, 0]), size=1)

        draw_robot(joints)

        if animation_name == 'IKtest':
            draw_T(T, color=[0.25, 0.25, .25])

        pygame.display.flip()
        pygame.time.wait(10)

        pygame_quit = False
        if pygame_quit: break


if __name__ == "__main__":
    import kinematics

    # pose_display = np.array([[1, 0, 0, 1 * SCALAR0],
    #                 [0, 1, 0, 1 * SCALAR0],
    #                 [0, 0, 1, 1 * SCALAR0],
    #                 [0, 0, 0, 1]])
    pose = np.array([[1,0,0,-206.58],
                     [0,1,0,0],
                     [0,0,1,144.21],
                     [0,0,0,1]])
    joints = np.array([0, 0, 0, 0, 0, 0])  # 0 position
    # joints2 = kinematics.IK(pose)
    display_robot(joints, animation_name='IKtest', T=pose)


# Deprecated -------------------------------------------------------------
"""
def displayTlist(Tlist, animation_name=''):
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glClearColor(1, 1, 1, 1)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, -2.0, -20)
    glRotatef(-60, 1, 0, 0)
    glRotatef(-30, 0, 0, 1)
    if animation_name[:-1] == 'jointsweep':
        direction = 1
        joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        param = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        #glRotatef(.1, 0, 0, 1)
        if animation_name == 'jointsweep1':
            Tlist = []
            if joints[param] > np.pi / 2:
                direction = -1
            elif joints[param] < 0:
                direction = 1
                param += 1
                param = param % 5
            joints[param] += (np.pi / 200) * direction
            for i in range(5):
                Tlist.append(kinematics.FK_dh(joints, i))
        elif animation_name == 'jointsweep2':
            Tlist = []
            if joints[param] > np.pi / 2:
                param += 1
                param = param % 5
            joints[param] += (np.pi / 200) * direction
            for i in range(5):
                Tlist.append(kinematics.FK_dh(joints, i))

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_axes(np.array([0, 0, 0]), np.array(
            [1, 0, 0]), np.array([0, 1, 0]), size=1)

        pygame.display.flip()
        pygame.time.wait(10)
"""
