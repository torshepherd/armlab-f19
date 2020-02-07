import numpy as np


class T:
    """Class T is for SE(3) matrix
    """
    def __init__(self, R, t):
        '''
        Specify homogenous transformation matrix by R (3x3) and p(3,)
        '''
        assert R.shape == (3,3), "R shape must be (3,3)"
        
        self.R = R
        if t.shape == (3,):
            self.t = t
        elif t.shape == (3,1):
            self.t = t.reshape(-1)
        else:
            raise Exception("t shape must be (3,) or (3,1)")
        
        self.matrix = np.vstack([np.hstack([self.R, self.t.reshape(-1,1)]), 
                                 np.array([0, 0, 0, 1])])
    
    @staticmethod
    def create(T_mat):
        assert T_mat.shape[1] == 4, "input must has 4 columns."
        return T(T_mat[:3,:3], T_mat[:3,-1])
        
    
    def __str__(self):
        print(self.matrix.astype(np.float16), end='')
        return ''
    
    
    def __getitem__(self, idx):
        return self.matrix[idx]
    
    
    def __setitem__(self, idx, val):
        self.matrix[idx] = val
    
    
    def inv(self):
        R = self.R.T
        t = -np.matmul(R, self.t.reshape(-1, 1)).reshape(-1)
        return T(R, t)
        
    
    @staticmethod
    def Inv(T_mat):
        assert isinstance(T_mat, T), "Input must be a T instance"
        return T_mat.inv()
    
    
    def __mul__(self, other):
        if not isinstance(other, T):
            if other.shape != (3,):
                raise Exception("Must be a T, or a vector of 3")
            else:
                other = np.append(other, 1)
                product = np.matmul(self.matrix, other.reshape(-1,1))
                return product[:3].reshape(-1).astype(np.float16)
        else:
            product = np.matmul(self.matrix, other.matrix)
            return T.create(product)
    
    
    @staticmethod
    def mm(A, B):
        return A*B


    def get_euler_angles(self):
        """
        TODO: implement this function
        return the Euler angles from a T matrix

        """
        # R = self.R
        # # theta_x = atan2(R32, R33)
        # # theta_y = atan2(-R31, sqrt(R32^2 + R33^2))
        # # theta_z = atan2(R21, R11)
        # theta_x = np.arctan2(R[2,1], R[2,2])
        # theta_y = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
        # theta_z = np.arctan2(R[1,0], R[0,0])
        # return np.array([theta_x, theta_y, theta_z])
        assert isRotationMatrix(self.R), "R is not a rotation matrix"
        
        sy = np.sqrt(self.R[0,0]**2 +  self.R[1,0]**2)
        singular = sy < 1e-6
    
        if not singular :
            x = np.arctan2(self.R[2,1] , self.R[2,2])
            y = np.arctan2(-self.R[2,0], sy)
            z = np.arctan2(self.R[1,0], self.R[0,0])
        else :
            x = np.arctan2(-self.R[1,2], self.R[1,1])
            y = np.arctan2(-self.R[2,0], sy)
            z = 0
    
        return np.array([x, y, z])


    def get_pose(self):
        """
        TODO: implement this function
        return the joint pose from a T matrix
        of the form (x,y,z,phi) where phi is rotation about base frame y-axis

        Note: why y-axis????
        """
        pos = self.t
        phi = np.arctan2(-T[2,0], np.sqrt(T[2,1]**2, T[2,2]**2))
        return np.append(pos, phi)
    
    
    def get_epsilon(self):
        phi = get_phi_from_R(self.R)
        J = get_J_in_SE(phi)
        rho = np.matmul(np.linalg.inv(J), self.t).reshape(-1)
        return np.concatenate((phi, rho))


def isRotationMatrix(R) :
    shouldBeIdentity = np.dot(R.T, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def to_s_matrix(w, v):
    """
    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)
    
    Siyuan's NOTES:
    Deprecated
    This func is replaced by the following ones.
    """
    pass


def get_T_from_euler_pos(euler, pos):
    R = get_R_from_euler_angles(euler)
    return T(R, pos)


def get_R_from_euler_angles(theta):
    # c_x = np.cos(euler[0])
    # s_x = np.sin(euler[0])
    # c_y = np.cos(euler[1])
    # s_y = np.sin(euler[1])
    # c_z = np.cos(euler[2])
    # s_z = np.sin(euler[2])
    # R = np.array([[c_y*c_z, c_z*s_x*s_y-c_x*s_z, s_x*s_z + c_x*c_z*s_y],
    #               [c_y*s_z, c_x*c_z+s_x*s_y*s_z, c_x*s_y*s_z-s_x*c_z],
    #               [-s_y, c_y*s_x, c_x*c_y]])
    R_x = np.array([[1,         0,                0                 ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]])
         
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                   1,      0                 ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]])
                 
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                   0,                    1]])
                     
    return np.dot(R_z, np.dot(R_y, R_x))


def vec2skew(vec):
    assert vec.shape == (3,), "Vector must be ndarray of shape (3,)"
    a = vec[0]
    b = vec[1]
    c = vec[2]
    return np.array([[0, -c, b],
                     [c, 0, -a],
                     [-b, a, 0]])


def get_R_from_phi(phi):
    """Get SO(3) matrix R from Lie algebra notation epsilon \in R^6
    Implement Rodregues Formula to transform rotation vector in R^3 to SO(3)
    """
    theta = np.linalg.norm(phi)
    theta = np.unwrap(theta) # wrap angle to [-pi, pi]
    dir_vec = phi / theta
    R_term_a = np.eye(3) * np.cos(theta)
    R_term_b = np.matmul(dir_vec.reshape(-1,1), 
                         dir_vec.reshape(1,-1)) * (1-np.cos(theta))
    R_term_c = vec2skew(dir_vec) * np.sin(theta)
    return R_term_a + R_term_b + R_term_c
    

def get_phi_from_R(R):
    """Notice that the returned value may has numerical deviation
    smaller than 1e-16. Making it float16 shall be okay (not implemented).
    """
    theta = np.arccos((np.trace(R) - 1) / 2)
    _, v = np.linalg.eig(R)
    imag = v.imag
    real = v.real
    for col in range(v.shape[1]):
        if np.count_nonzero(imag[:,col]) == 0:
            return real[:,col].reshape(-1) * theta
    raise Exception("No valid real eigenvectors found.")


def get_J_in_SE(phi):
    """ J*rho = translation vector t in SE(3)
    """
    theta = np.linalg.norm(phi)
    theta = np.unwrap(theta) # wrap angle to [-pi, pi]
    dir_vec = phi / theta
    J_term_a = np.sin(theta) / theta * np.eye(3)
    J_term_b = np.matmul(dir_vec.reshape(-1,1), 
                         dir_vec.reshape(1,-1)) * (1 - np.sin(theta) / theta)
    J_term_c = vec2skew(dir_vec) * np.cos(theta) / theta
    return J_term_a + J_term_b + J_term_c


def get_T_from_epsilon(epsilon):
    """Get SE(3) matrix T from Lie algebra se(3) epsilon \in R^6
    - epsilon: ndarray shape (6,)
        - [phi, rho] by default
        - phi: rotation vector
        - rho: translation vector
    """
    phi = epsilon[:3]
    rho = epsilon[3:]
    R = get_R_from_phi(phi)
    J = get_J_in_SE(phi)
    t = np.matmul(J, rho.reshape(-1,1))
    return T(R, t)


if __name__ == "__main__":
    # R = np.eye(3)
    # t = np.arange(3)
    # a = T(R, t)
    # print(a[:3,:3])
    # print(a.inv())
    # print(a*a)
    theta = np.array([np.pi/2, 0, 0])
    R = get_R_from_euler_angles(theta)
    Tmat = T(R, np.zeros(3))
    euler = Tmat.get_euler_angles()
    print(theta)
    print(R)
    print(euler)
    