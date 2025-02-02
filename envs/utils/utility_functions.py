import numpy as np

def rot_x(theta):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])
    return Rx

def rot_y(theta):
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    return Ry

def rot_z(theta):
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    return Rz

def rot_2d(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s],
                     [s, c]]).reshape(2,2)

def skew(v):
    S = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return S

def time_derivative(f,qq,qqd,xx,xxd,t):
    dfdt = np.diff(f.subs(qq,xx), t).subs(xxd,qqd).subs(xx,qq)
    return dfdt