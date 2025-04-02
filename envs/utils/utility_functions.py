import numpy as np
import matplotlib.pyplot as plt


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

def dual_tanh(x):
    mg = 8.19135  # if mini: 2.7468
    max_thrust = 30.0  # if mini: 15.7837
    tanh1 = mg * (1 + np.tanh(3 * x + 2)) / 2
    tanh2 = (max_thrust - mg) * (1 + np.tanh(3 * x - 2)) / 2
    return tanh1 + tanh2

def dual_tanh_payload(x):
    mg = 8.28945  # if mini: 2.7468
    max_thrust = 30.0  # if mini: 15.7837
    tanh1 = mg * (1 + np.tanh(3 * x + 2)) / 2
    tanh2 = (max_thrust - mg) * (1 + np.tanh(3 * x - 2)) / 2
    return tanh1 + tanh2

def dual_tanh_tvec(x):
    mg = 8.19135  # if mini: 2.7468
    max_Fz = 15.0  # if mini: 15.7837
    tanh1 = mg * (1 + np.tanh(3 * x + 2)) / 2
    tanh2 = (max_Fz - mg) * (1 + np.tanh(3 * x - 2)) / 2
    return tanh1 + tanh2

if __name__ == "__main__":

    x_vals = np.linspace(-1, 1, 500)
    y_vals = dual_tanh_tvec(x_vals)

    plt.plot(x_vals, y_vals)
    plt.xlabel("x")
    plt.ylabel("dual_tanh_tvec(x)")
    plt.title("Graph of dual_tanh_tvec from x = -1 to x = 1")
    plt.grid(True)
    plt.show()