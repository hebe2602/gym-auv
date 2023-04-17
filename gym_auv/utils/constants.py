import numpy as np


m = 23.8
x_g = 0.046
I_z = 1.760
X_udot = -2.0
Y_vdot = -10.0
Y_rdot = 0.0
N_rdot = -1.0
N_vdot = 0.0
X_u = -0.7225
Y_v = -0.8612
Y_r = 0.1079
N_v = 0.1052
N_r = -0.5
X_uu = -1.3274
Y_vv = -36.2823
Y_rr = -0.02
Y_rv = -0.01
Y_vr = -0.01
N_vv = 5.0437
N_rr = 0.005
N_rv = -0.001
N_vr = -0.001
Y_uvb = -0.5*1000*np.pi*1.24*(0.15/2)**2
Y_uvf = -1000*3*0.0064
Y_urf = -0.4*Y_uvf
N_uvb = (-0.65*1.08 + 0.4)*Y_uvb
N_uvf = -0.4*Y_uvf
N_urf = -0.4*Y_urf
Y_uudr = 19.2
N_uudr = -0.4*Y_uudr
X_uuu = -5.8664
m_11 = m - X_udot
m_22 = m-Y_v
m_23 = m*x_g - Y_rdot
m_32 = m*x_g - N_vdot
m_33 = I_z - N_rdot

MAX_SPEED = 2

M =  np.array([[m - X_udot, 0, 0],
    [0, m - Y_vdot, m*x_g - Y_rdot],
    [0, m*x_g - N_vdot, I_z - N_rdot]]
)   
M_inv = np.linalg.inv(M)
 

B = np.array([
        [1, 0],
        [0, -1.7244],
        [0, 1],
    ])

def C(nu):
    u = nu[0]
    v = nu[1]
    r = nu[2]

    c_13 = -M[0,0]*v - M[1,2]*r
    c_23 = M[0,0]*u

    C = np.array([
        [0,                 0,         c_13],
        [0,                 0,         c_23],
        [-c_13,            -c_23,      0]
    ])  
    return C

def D(nu):
    u = nu[0]
    v = nu[1]
    r = nu[2]

    d_11 = -X_u - X_uu * np.abs(u) - X_uuu * u**2
    d_22 = -Y_v - Y_vv * np.abs(v) - Y_rv * np.abs(r)
    d_23 = -Y_r - Y_vr * np.abs(v) - Y_rr * np.abs(r)
    d_32 = -N_v - N_vv * np.abs(v) - N_rv * np.abs(r)
    d_33 = -N_r - N_vr * np.abs(v) - N_rr * np.abs(r)

    D = np.array([
        [d_11,  0,    0],
        [0,     d_22, d_23],
        [0,     d_32, d_33]
    ])
    return D

def N(nu):
    u = nu[0]
    v = nu[1]
    r = nu[2]
    N = np.array([
        [2.0, 0, 0],
        [0, 7.0, m*u + 0.1],
        [0, 0.1, m*x_g*u + 0.5]
    ])  
    return N
