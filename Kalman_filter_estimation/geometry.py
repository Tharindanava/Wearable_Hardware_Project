"""
geometry.py
20/04/2024

Implements functions related to vectors, rotations, and quaternions.
"""

import numpy as np
import helper_functions as hf
from scipy.linalg import expm
from helper_functions import list2vec as l2v
from numpy.linalg import norm
from scipy.spatial.transform import rotation as scirot

#region Independent definitions on vectors/matrices

eye3 = np.identity(3)
eye6 = np.identity(6)

def normalize_vector(vec):
    return vec/norm(vec)
#enddef

def angle_between(v1, v2):
    return np.dot(v1.transpose(), v2)/(norm(v1)*norm(v2))
    pass
#enddef

def sqrm(mat):
    return mat @ mat
#enddef

#endregion

#region Independent definitions on quaternions

# Defines the identity quaternion w.r.t multiplication
identity_quaternion = l2v([0,0,0,1])

# Standard rotation matrices (about x, y, z axes)
def R1(angle):
    return l2v([1.0,            0.0,            0.0],
               [0.0,  np.cos(angle), -np.sin(angle)],
               [0.0, -np.sin(angle),  np.cos(angle)])
#enddef
def R2(angle):
    return l2v([ np.cos(angle), 0.0, np.sin(angle)],
               [           0.0, 1.0,           0.0],
               [-np.sin(angle), 0.0, np.cos(angle)])
#enddef
def R3(angle):
    return l2v([ np.cos(angle), -np.sin(angle), 0.0],
               [ np.sin(angle),  np.cos(angle), 0.0],
               [           0.0,            0.0, 1.0])
#enddef

# Returns the 3x3 cross product matrix of the given 3x1 vector 'vector'.
def cross_prod_mat_3(vector):
    v1 = vector[0,0]
    v2 = vector[1,0]
    v3 = vector[2,0]
    
    return np.array([[0, -v3, v2],
                     [v3, 0, -v1],
                     [-v2, v1, 0]])
#enddef

# Returns the inverse of the 3x3 cross product matrix 'mat' (a vector)
def inverse_hat_map_3(mat):
    v1 = mat[2,1]
    v2 = mat[0,2]
    v3 = mat[1,0]
    return l2v([v1,v2,v3])
#enddef

# Returns the 4x4 cross product matrix of the given 3x1 vector 'vector'.
def cross_prod_mat_4(vector):
    v1 = vector[0,0]
    v2 = vector[1,0]
    v3 = vector[2,0]

    return np.array([[  0,  v3, -v2, v1],
                     [-v3,   0,  v1, v2],
                     [ v2, -v1,   0, v3],
                     [-v1, -v2, -v3,  0]])
#enddef

# Returns the curl L map of the quaternion 'q'.
def quaternion_curl_L_map(q):
    q1, q2, q3, q4 = q[0,0], q[1,0], q[2,0], q[3,0]
    return np.array([[ q4,  q3, -q2, q1],
                     [-q3,  q4,  q1, q2],
                     [ q2, -q1,  q4, q3],
                     [-q1, -q2, -q3, q4]])
#enddef

# Returns the curl R map of the quaternion 'q'.
def quaternion_curl_R_map(q):
    q1, q2, q3, q4 = q[0,0], q[1,0], q[2,0], q[3,0]
    return np.array([[ q4, -q3,  q2, q1],
                     [ q3,  q4, -q1, q2],
                     [-q2,  q1,  q4, q3],
                     [-q1, -q2, -q3, q4]])
#enddef

# Returns the Hamiltonian product q1*q2 of two quaternions 'q1' and 'q2'.
# This corresponds to 'sum' of rotations: first the rotation q2, then the rotation q1.
def quaternion_multiply(q1, q2):
    q11, q12, q13, q14 = q1[0,0], q1[1,0], q1[2,0], q1[3,0]
    q21, q22, q23, q24 = q2[0,0], q2[1,0], q2[2,0], q2[3,0]

    return np.array([[ q14*q21+q13*q22-q12*q23+q11*q24],
                     [-q13*q21+q14*q22+q11*q23+q12*q24],
                     [ q12*q21-q11*q22+q14*q23+q13*q24],
                     [-q11*q21-q12*q22-q13*q23+q14*q24]])
    return np.dot(quaternion_curl_L_map(q1), q2)
    return np.dot(quaternion_curl_R_map(q2), q1)
#enddef

# Returns the inverse of the given quaternion 'q'.
def quaternion_inverse(q):
    q1, q2, q3 = q[0,0], q[1,0], q[2,0]
    q4 = q[3,0]
    return l2v([-q1, -q2, -q3, q4])
#enddef

#endregion

#region Dependent definitions on quaternions/rotations

# Returns the matrix exponential of matrix 'mat'.
# Exact alias for scipy.linalg.expm()
def mat_exp(mat):
    return expm(mat)
#enddef

# Returns the 3x3 cross product matrix of the given 3x1 vector 'vector'.
# Exact alias to cross_prod_mat_3(vector).
def hat_map_3(vector):
    return cross_prod_mat_3(vector)
#enddef

# Returns the cross product between the two vectors 'v1' and 'v2'.
# Functionality same as numpy.cross(a, b).
def vector_cross_product(v1, v2):
    return np.dot(cross_prod_mat_3(v1), v2)
#enddef

# Returns a quaternion representing the given the axis of rotation 'axis' (3x1 vector),
# and the angle of rotation 'angle'.
def axis_angle_to_quaternion(axis, angle):
    k1, k2, k3 = axis[0,0], axis[1,0], axis[2,0]
    q1 = k1*np.sin(angle/2.0)
    q2 = k2*np.sin(angle/2.0)
    q3 = k3*np.sin(angle/2.0)
    q4 =    np.cos(angle/2.0)
    return np.block([q1, q2, q3, q4]).transpose()
#enddef

# Returns the rotation matrix represeting the given 'axis' and 'angle'.
def axis_angle_to_rotation_matrix(axis, angle):
    return expm(angle*hat_map_3(axis))
#enddef

# Returns the rotation matrix form of the given quaternion 'q'.
def quaternion_to_rotation_matrix(q):
    axis = q[0:3,:]
    q4 = q[3,0]
    axis_hat = hat_map_3(axis)
    return np.identity(3) + 2*q4*axis_hat + 2*np.dot(axis_hat, axis_hat)
#enddef

# Returns the axis and angle represented by the quaternion 'q'.
def quaternion_to_axis_angle(q):
    q4 = q[3,0]
    angle = 2*np.arccos(q4)
    axis = None
    if angle == 0.0:
        axis = q[0:3,:]
    else:
        axis = q[0:3,:]/np.sin(angle/2.0)
    return axis, angle
#enddef

# Returns the quaternion form of the given rotation matrix 'rot_mat'
def rotation_matrix_to_quaternion(rot_mat):
    axis, angle = rotation_matrix_to_axis_angle(rot_mat)
    return axis_angle_to_quaternion(axis, angle)
#enddef

# Returns the axis and angle represented by the rotation matrix 'rot_mat'.
def rotation_matrix_to_axis_angle(rot_mat):
    angle = np.arccos(0.5*(np.trace(rot_mat)-1.0))
    axis = inverse_hat_map_3((rot_mat - rot_mat.transpose())/(4.0*np.cos(angle/2.0)))/np.sin(angle/2.0)
    return axis, angle
#enddef

# Returns the zyx Euler angle representation of the given matrix 'mat'.
def rotm2eul(mat, scheme='xyz'):
    rot_mat = scirot.Rotation.from_matrix(mat)
    zyx_euler_angles = rot_mat.as_euler(scheme, degrees=True)
    zyx_euler_angles = zyx_euler_angles.reshape((3,1))
    return zyx_euler_angles
#enddef

#endregion

#region Applications of rotations

# Applies the given rotation 'rot_mat' to 'vector',
# then translates the result by 'trans_vector', and returns the resulting vector.
def rotate_translate_body(vector, rot_mat, trans_vector):
    return np.dot(rot_mat, vector) + trans_vector
#enddef

#endregion
