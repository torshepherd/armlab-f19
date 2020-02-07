import numpy as np
from math import sqrt
# expm is a matrix exponential function
from scipy.linalg import expm
from kinematics_convert import T
import kinematics_convert as kc

""" 
TODO: Here is where you will write all of your kinematics functions 
There are some functions to start with, you may need to implement a few more

"""

def warpToPi(angles):

    angles %= 2*np.pi
    idx = angles > np.pi
    angles[idx] = angles[idx] - 2*np.pi
    return angles

# DH table:
p2 = np.pi / 2
alphas = [p2, 0, p2, -p2, p2, 0]
# a_list = [0, 100, 0, 0, 50]
# d_list = [117, 0, 0, 110, 0]
a_list = [0, 100, 0, 0, 0, 36] # or -36
d_list = [117, 0, 0, 110, 0, 80] # 80 can be decreased
links = [117, 100, 0, 110, 0, sqrt(a_list[-1]**2 + d_list[-1]**2)]

correct_offset = np.array([0, np.pi/2, np.pi/2, 0, np.pi/4, np.pi/4])
std_offset = np.array([0, -np.pi, np.pi/2, 0, np.pi/2, 0])
add_offset = warpToPi(correct_offset - std_offset)


def theta_dh(joint_angles):
    '''
    TODO: experimentally determine offsets and multipliers to
    convert the motor encoder readings into DH-specified angles
    '''    
    return joint_angles + std_offset


def FK_dh(joint_angles, link):
    """
    TODO: implement this function

    Calculate forward kinematics for rexarm using DH convention

    return a transformation matrix representing the pose of the 
    desired link

    note: phi is the euler angle about the y-axis in the base frame

    """

    '''
    TOR'S NOTES:

    ex = {{r, r, r, r, r}, {0, 1.00, 0, 0, .5}, {Pi/2, 0, Pi/2, -Pi/2, 
    Pi/2}, {1.17, 0, 0, 1.10, 0}, {q1, q2, q3, q4, q5}};

    Siyuan's DH table:
    n|  a|  d| alpha| theta
    1|  0|  3|  pi/2|theta0
    2|  1|  0|     0|theta1 + pi
    3|  0|  0|  pi/2|theta2 + pi/2
    4|  0|  2| -pi/2|theta3
    5|  0|  0|  pi/2|theta4 + pi/2
    6| -1|  1|     0|theta5

    The function should return T_0_i where i is the number of the origin specified by link
    
    '''

    # Convert to DH angles
    thetas = theta_dh(joint_angles)

    # Construct T_n-1_n matrices up to link
    T_0_link = T.create(np.eye(4))

    for i in range(link):
        c_t = np.cos(thetas[i])
        s_t = np.sin(thetas[i])

        c_a = np.cos(alphas[i])
        s_a = np.sin(alphas[i])

        d = d_list[i]
        a = a_list[i]

        A_i_R = np.array([[c_t, -1 * s_t * c_a, s_t * s_a],
                        [s_t, c_t * c_a, -1 * c_t * s_a],
                        [0, s_a, c_a]])
        A_i_t = np.array([a * c_t, a * s_t, d])
        
        A_i = T(A_i_R, A_i_t)

        T_0_link = T_0_link * A_i
        
    print("pos_{}: ".format(link), T_0_link.t.astype(np.float16))

    return T_0_link


def FK_pox(joint_angles):
    """
    TODO: implement this function

    Calculate forward kinematics for rexarm
    using product of exponential formulation

    return a 4-tuple (x, y, z, phi) representing the pose of the 
    desired link

    note: phi is the euler angle about y in the base frame

    """
    pass


def IK(pose):
    """
    TODO: implement this function

    Calculate inverse kinematics for rexarm

    return the required joint angles

    - Siyuan's NOTES:
    with the pose of the end effect, obtain the 5th joint spatial position first.
    then, deduce 1st joint angle; 3rd angle; 2nd angle; 5th angle; 
    4th angle; 6th angle in this sequence.
    
    - Siyuan's naming convention:
    
    
    """
    T_06 = pose
    
    # Get 3D position for o_c
    # o_4 = [I|t]*pose, where t is [a_6 0 -d_6-l_5]
    T_65_t = np.array([-a_list[-1], 0, -d_list[-1]])
    T_65 = T(np.eye(3), T_65_t)
    T_05 = T_06*T_65
    coord_4 = T_05.t
    
    l1 = links[0]
    l2 = links[1]
    l4 = links[2] + links[3]
    xc = coord_4[0]
    yc = coord_4[1]
    zc = coord_4[2]
    ################################# Correct Line
    rc = np.sqrt(xc**2 + yc**2)
    l14 = np.sqrt(rc**2 + (zc-l1)**2)
    
    q1 = np.arctan2(yc, xc) - np.pi
    q3 = np.arccos(((l14**2-l2**2-l4**2)/(2*l2*l4)).astype(np.float16).astype(np.float64))
    alpha = -np.arccos(((l4**2-l2**2-l14**2)/(2*l2*l14)).astype(np.float16).astype(np.float64))
    beta = np.arcsin(((zc-l1)/l14).astype(np.float16))
    q2 = -alpha - beta + np.pi    

    T_03 = FK_dh([q1, q2, q3, 0, 0, 0], 3)
    T_36 = T_03.inv() * T_06
    
    # euler_36 = T_36.get_euler_angles()
    # print("euler_36: ", euler_36.astype(np.float16))

    R_36 = T_36.R
    q4 = np.arctan2(R_36[1,2], R_36[0,2])
    q5 = np.arctan2(np.sqrt(1-R_36[2,2]**2), R_36[2,2]) - np.pi/2
    q6 = np.arctan2(R_36[2,1], -R_36[2,0])
    return warpToPi(np.array([q1, q2, q3, q4, q5, q6]))
    
if __name__ == "__main__":
    """ rot_vec = np.array([0,1,2])
    R = get_R_from_phi(rot_vec)
    v = get_phi_from_R(R)
    print(v) """
    
    
    import vis
    # pose = np.array([[1,0,0,-400],
    #                  [0,1,0,0],
    #                  [0,0,1,100],
    #                  [0,0,0,1]])
    # print("add offsets: ", add_offset)
    
    # motor_joints = np.array([np.pi/6, -np.pi/6, +np.pi/6, np.pi/6, np.pi/6, np.pi/6])  # 0 position
    # joints = warpToPi(motor_joints + add_offset)
    # # joints = np.zeros(6)
    # print("joints with ", joints)

    # pose_06 = FK_dh(joints, 6)

    # joints2 = warpToPi(IK(pose_06) - add_offset)
    # joints2 = IK(pose_06)
    # print("Predicted Joints", joints2.astype(np.float16))
    
    # vis.display_robot(joints, animation_name='IKtest', T=pose_06)
    
    
    motor_joints = np.array([0, 0.9774, 1.065, 0, 0.314, -0.89])
    joints = warpToPi(motor_joints + add_offset)
    print("joints with ", joints)

    pose_06 = FK_dh(joints, 6)
    print(pose_06)

    # joints2 = warpToPi(IK(pose_06) - add_offset)
    joints2 = IK(pose_06)
    print("Predicted Joints", joints2.astype(np.float16))
    
    vis.display_robot(joints, animation_name='IKtest', T=pose_06)