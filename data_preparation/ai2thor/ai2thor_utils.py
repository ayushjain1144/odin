import torch
import numpy as np

import ipdb
st = ipdb.set_trace


def eul2rotm_py(rx, ry, rz):
    rx = rx[:,np.newaxis]
    ry = ry[:,np.newaxis]
    rz = rz[:,np.newaxis]
    # these are B x 1
    sinz = np.sin(rz)
    siny = np.sin(ry)
    sinx = np.sin(rx)
    cosz = np.cos(rz)
    cosy = np.cos(ry)
    cosx = np.cos(rx)
    r11 = cosy*cosz
    r12 = sinx*siny*cosz - cosx*sinz
    r13 = cosx*siny*cosz + sinx*sinz
    r21 = cosy*sinz
    r22 = sinx*siny*sinz + cosx*cosz
    r23 = cosx*siny*sinz - sinx*cosz
    r31 = -siny
    r32 = sinx*cosy
    r33 = cosx*cosy
    r1 = np.stack([r11,r12,r13],axis=2)
    r2 = np.stack([r21,r22,r23],axis=2)
    r3 = np.stack([r31,r32,r33],axis=2)
    r = np.concatenate([r1,r2,r3],axis=1)
    return r

def get_origin_T_camX(event):
    position = np.array(list(event.metadata["agent"]["position"].values())) + np.array([0.0, 0.675, 0.0]) # adjust for camera height from agent
    rotation = np.array(list(event.metadata["agent"]["rotation"].values()))

    rx = -np.radians(event.metadata["agent"]["cameraHorizon"]) # pitch
    ry = np.radians(rotation[1]) # yaw
    rz = 0. # roll is always 0
    rotm = eul2rotm_py(np.array([rx]), np.array([ry]), np.array([rz]))
    origin_T_camX = np.eye(4)
    origin_T_camX[0:3,0:3] = rotm
    origin_T_camX[0:3,3] = position

    world_t_weird = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0,-1, 0, 0],
            [0, 0, 0, 1]
        ],
        dtype=np.float32
    )

    origin_T_camX = world_t_weird @ origin_T_camX
    origin_T_camX = torch.from_numpy(origin_T_camX)
    return origin_T_camX


