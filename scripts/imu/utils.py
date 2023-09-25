import numpy as np

def quaternConj(q):
    qConj = np.array([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]]).T
    return qConj

def quaternRotate(v, q):
    row, col = v.shape
    zero_column = np.zeros((row, 1))
    v0XYZ = quaternProd(quaternProd(q, np.hstack((zero_column, v))), quaternConj(q))
    v = v0XYZ[:, 1:4]
    return v


def quaternProd(a, b):
    ab = np.zeros((a.shape[0], 4))
    ab[:, 0] = a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1] - a[:, 2] * b[:, 2] - a[:, 3] * b[:, 3]
    ab[:, 1] = a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0] + a[:, 2] * b[:, 3] - a[:, 3] * b[:, 2]
    ab[:, 2] = a[:, 0] * b[:, 2] - a[:, 1] * b[:, 3] + a[:, 2] * b[:, 0] + a[:, 3] * b[:, 1]
    ab[:, 3] = a[:, 0] * b[:, 3] + a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1] + a[:, 3] * b[:, 0]
    return ab