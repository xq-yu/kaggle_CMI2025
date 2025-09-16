import numpy as np
import numpy as np
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings("ignore")



def quaternion_to_rotation(quat):
    """
    transform quaternion to rotation matrix
    Args:
        quat:[N,rot_x,rot_y,rot_z, rot_w]
    
    return:
        np.array: rotation matrix [N,9]
    """
    # 确保输入是NumPy数组
    quat = np.asarray(quat, dtype=np.float64)
        
    # 创建Rotation对象
    rotation = R.from_quat(quat)
    return rotation.as_matrix().reshape(-1,9)


def sep_gravity(acc,quat):
    '''
    separate gravity from acceleration
    params:
        acc: [N,acc_x,acc_y,acc_z]
        quat:[N,rot_x,rot_y,rot_z, rot_w]
    output
        acc_new:[N,acc_x,acc_y,acc_z]
        gravity:[N,g_x,g_y,g_z]
    '''
    rot = R.from_quat(quat)
    rot_reverse = rot.inv()
    gravity = rot_reverse.apply([0,0,9.8])
    acc_new = acc-gravity
    return acc_new,gravity

def quaternion_to_euler(quat):
    # [N x,y,z,w]
    euler_angles = np.zeros((len(quat),3))*np.nan
    nan_idx = np.isnan(quat[:,0]) | np.isnan(quat[:,1]) | np.isnan(quat[:,2]) | np.isnan(quat[:,3])

    rotation = R.from_quat(quat[~nan_idx,:])
    euler_angles[~nan_idx,:] = rotation.as_euler('zyx', degrees=False) 
    euler_z,euler_y,euler_x = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]

    return euler_x, euler_y, euler_z  


def euler_to_quaternion(euler_x, euler_y, euler_z):
    rotation = R.from_euler('zyx', np.array([euler_z, euler_y, euler_x]).T, degrees=False)
    q = rotation.as_quat()  # [x, y, z, w]
    return np.array([q[:,3],q[:,0],q[:,1],q[:,2]]).T  # [w, x,y, z]



def quaternion_multiply(q2, q1):
    '''
    args:
        q1,q2: [N,[w,x,y,z]]
    return:
        [N,[w,x,y,z]]
    '''

    if len(q2.shape)==1:
        w2, x2, y2, z2 = q2[0],q2[1],q2[2],q2[3]
    else:
        w2, x2, y2, z2 = q2[:,0],q2[:,1],q2[:,2],q2[:,3]

    if len(q1.shape)==1:
        w1, x1, y1, z1 = q1[0],q1[1],q1[2],q1[3]
    else:
        w1, x1, y1, z1 = q1[:,0],q1[:,1],q1[:,2],q1[:,3]
    return np.array([
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1
    ]).T


def quaternion_conjugate(q):
    if len(q.shape)==1:
        w, x, y, z = q[0],q[1],q[2],q[3]
    else:
        w, x, y, z = q[:,0],q[:,1],q[:,2],q[:,3]
    return np.array([w, -x, -y, -z]).T


def relative_rotation_quaternion(q1, q2, local):
    '''
    q1,q2: [N,[w,x,y,z]]
    '''
    q1_conjugate = quaternion_conjugate(q1)
    if local:
        q_rel = quaternion_multiply(q1_conjugate,q2)
    else:
        q_rel = quaternion_multiply(q2, q1_conjugate)
        
    return q_rel

