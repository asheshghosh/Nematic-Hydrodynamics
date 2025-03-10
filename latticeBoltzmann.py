"""
Author: Ashesh Ghosh
Last Update: Feb 12 2025
Description: Optimized simulation code for active nematic in 2d.
"""
import numpy as np
import math
import random
from numba import njit, prange # type: ignore

# --------------------------
# Parameters (from parameters.c)
# --------------------------
FILE_NAME = "active_nematic"

I = 100
J = 100
NMAX = I * J

TIME_STEPS = 10000
TIME_WRITE = 100

LATTICE_VELOCITY_NUMBER = 9
STPROC = 2  # (not used directly; parallelism is handled by Numba)

C = 1.0
DT = 0.01
TAUF = 2.0 * DT
DENSITYINIT = 2.0 / DT
LAMBDA = 1.1
ALPHA = 1.00

LMARKBULK = 2    # bulk
LMARKBC = 4      # boundary condition

# --------------------------
# Numba-friendly utility functions
# --------------------------
@njit(inline='always')
def i_vr(l):
    return l % I

@njit(inline='always')
def j_vr(l):
    return l // I

# --------------------------
# Initialization of lattice vectors and weights
# --------------------------
@njit
def initialise_E():
    E = np.zeros((LATTICE_VELOCITY_NUMBER, 2), dtype=np.int64)
    WKONST = np.zeros(LATTICE_VELOCITY_NUMBER, dtype=np.float64)
    E[0, 0] = 0;  E[0, 1] = 0
    E[1, 0] = 1;  E[1, 1] = 0
    E[2, 0] = 0;  E[2, 1] = 1
    E[3, 0] = -1; E[3, 1] = 0
    E[4, 0] = 0;  E[4, 1] = -1
    E[5, 0] = 1;  E[5, 1] = 1
    E[6, 0] = -1; E[6, 1] = 1
    E[7, 0] = -1; E[7, 1] = -1
    E[8, 0] = 1;  E[8, 1] = -1

    WKONST[0] = 4.0 / 9.0
    for i in range(1, 5):
        WKONST[i] = 1.0 / 9.0
    for i in range(5, 9):
        WKONST[i] = 1.0 / 36.0
    return E, WKONST

# --------------------------
# LB functions
# --------------------------
@njit(parallel=True)
def computeFeq(U, FEQ, WKONST, LMARK, E):
    for l in prange(NMAX):
        if LMARK[l] != LMARKBULK:
            continue
        u2 = U[1, l]**2 + U[2, l]**2
        for m in range(LATTICE_VELOCITY_NUMBER):
            ue = U[1, l] * E[m, 0] + U[2, l] * E[m, 1]
            FEQ[m, l] = WKONST[m] * U[0, l] * (1.0 + 3.0 * ue + 4.5 * ue * ue - 1.5 * u2)

@njit(inline='always')
def calcLBlnew(l, m, E):
    i_coord = i_vr(l) + E[m, 0]
    j_coord = j_vr(l) + E[m, 1]
    return i_coord + j_coord * I

@njit(inline='always')
def calcLpbc(l):
    xp2 = 0
    yp2 = 0
    if i_vr(l) == 0:
        xp2 = I - 2
    if i_vr(l) == I - 1:
        xp2 = -(I - 2)
    if j_vr(l) == 0:
        yp2 = I * (J - 2)
    if j_vr(l) == J - 1:
        yp2 = -I * (J - 2)
    return l + xp2 + yp2

@njit(parallel=True)
def calcFNEW2F(F, FNEW, LMARK):
    for l in prange(NMAX):
        if LMARK[l] == LMARKBULK:
            for m in range(LATTICE_VELOCITY_NUMBER):
                F[m, l] = FNEW[m, l]

@njit
def calcF2U(l, U, F, E, LMARK):
    if LMARK[l] == LMARKBULK:
        density = 0.0
        fex = 0.0
        fey = 0.0
        for m in range(LATTICE_VELOCITY_NUMBER):
            density += F[m, l]
            fex += F[m, l] * E[m, 0]
            fey += F[m, l] * E[m, 1]
        U[1, l] = fex / density
        U[2, l] = fey / density
        U[0, l] = density
    else:
        U[0, l] = DENSITYINIT
        U[1, l] = 0.0
        U[2, l] = 0.0

@njit(parallel=True)
def compute_sigma(SIGMA, H, Q, LMARK):
    for l in prange(NMAX):
        if LMARK[l] != LMARKBULK:
            continue
        SIGMA[0, l] = -LAMBDA * H[0, l] + ALPHA * Q[0, l]
        SIGMA[1, l] = (-LAMBDA * H[1, l] +
                       Q[0, l] * H[1, l] - Q[1, l] * H[0, l] -
                       (H[0, l] * Q[1, l] - H[1, l] * Q[0, l]) +
                       ALPHA * Q[1, l])
        SIGMA[2, l] = (-LAMBDA * H[1, l] +
                       Q[1, l] * H[0, l] - Q[0, l] * H[1, l] -
                       (H[1, l] * Q[0, l] - H[0, l] * Q[1, l]) +
                       ALPHA * Q[1, l])
        SIGMA[3, l] = LAMBDA * H[0, l] - ALPHA * Q[0, l]
    for l in prange(NMAX):
        lpbc = calcLpbc(l)
        if lpbc != l:
            for m in range(2):
                SIGMA[m, l] = SIGMA[m, lpbc]

@njit(parallel=True)
def computeP(P, SIGMA, U, WKONST, LMARK, E, Q, H):
    compute_sigma(SIGMA, H, Q, LMARK)
    for l in prange(NMAX):
        if LMARK[l] != LMARKBULK:
            continue
        forceX = ((SIGMA[0, l+1] - SIGMA[0, l-1]) / 2.0 +
                  (SIGMA[1, l+I] - SIGMA[1, l-I]) / 2.0)
        forceY = ((SIGMA[2, l+1] - SIGMA[2, l-1]) / 2.0 +
                  (SIGMA[3, l+I] - SIGMA[3, l-I]) / 2.0)
        uF = U[1, l] * forceX + U[2, l] * forceY
        for m in range(LATTICE_VELOCITY_NUMBER):
            ue = U[1, l] * E[m, 0] + U[2, l] * E[m, 1]
            eF = E[m, 0] * forceX + E[m, 1] * forceY
            P[m, l] = (1.0 - DT / (2.0 * TAUF)) * WKONST[m] * (3.0 * eF - 3.0 * uF + 9.0 * ue * eF)

@njit(parallel=True)
def computePBC_LB(P, FEQ, F, LMARK):
    for l in prange(NMAX):
        if LMARK[l] & LMARKBC:
            lpbc = calcLpbc(l)
            for m in range(LATTICE_VELOCITY_NUMBER):
                P[m, l] = P[m, lpbc]
                FEQ[m, l] = FEQ[m, lpbc]
                F[m, l] = F[m, lpbc]

@njit(parallel=True)
def compute_LB_step(U, F, FNEW, FEQ, P, LMARK, WKONST, E, SIGMA, Q, H):
    computeFeq(U, FEQ, WKONST, LMARK, E)
    computeP(P, SIGMA, U, WKONST, LMARK, E, Q, H)
    computePBC_LB(P, FEQ, F, LMARK)
    for l in prange(NMAX):
        for m in range(LATTICE_VELOCITY_NUMBER):
            lnew = calcLBlnew(l, m, E)
            if lnew >= NMAX or lnew < 0:
                continue
            FNEW[m, lnew] = F[m, l] - DT / TAUF * (F[m, l] - FEQ[m, l]) + P[m, l]
    calcFNEW2F(F, FNEW, LMARK)
    for l in prange(NMAX):
        calcF2U(l, U, F, E, LMARK)
    for l in prange(NMAX):
        lpbc = calcLpbc(l)
        for m in range(3):
            U[m, l] = U[m, lpbc]

# --------------------------
# Finite Difference functions for Q-tensor evolution
# --------------------------
@njit(inline='always')
def compute_Q_laplacian(l, Q, Q_laplacian):
    lxp = l + 1
    lxm = l - 1
    lyp = l + I
    lym = l - I
    for m in range(2):
        Q_laplacian[m] = Q[m, lxp] + Q[m, lxm] + Q[m, lyp] + Q[m, lym] - 4 * Q[m, l]

@njit(inline='always')
def compute_u1(l, Q, Q_laplacian, u1, H):
    temp = 4 * (Q[0, l]**2 + Q[1, l]**2)
    u1[0] = Q_laplacian[0] - 2 * C * Q[0, l] * (temp - 1.0)
    u1[1] = Q_laplacian[1] - 2 * C * Q[1, l] * (temp - 1.0)
    H[0, l] = u1[0]
    H[1, l] = u1[1]

@njit(inline='always')
def compute_u2(l, U, u2):
    uxx = (U[1, l+1] - U[1, l-1]) / 2.0
    uxy = 0.5 * (((U[1, l+I] - U[1, l-I]) / 2.0) + ((U[2, l+1] - U[2, l-1]) / 2.0))
    u2[0] = LAMBDA * uxx
    u2[1] = LAMBDA * uxy

@njit(inline='always')
def compute_u3(l, U, Q, u3):
    dQxxdx = (Q[0, l+1] - Q[0, l-1]) / 2.0
    dQxydx = (Q[1, l+1] - Q[1, l-1]) / 2.0
    dQxxdy = (Q[0, l+I] - Q[0, l-I]) / 2.0
    dQxydy = (Q[1, l+I] - Q[1, l-I]) / 2.0
    u3[0] = -(U[1, l] * dQxxdx + U[2, l] * dQxxdy)
    u3[1] = -(U[1, l] * dQxydx + U[2, l] * dQxydy)

@njit(parallel=True)
def calcQNEW2Q(Q, QNEW, LMARK):
    for l in prange(NMAX):
        lpbc = calcLpbc(l)
        for m in range(2):
            Q[m, l] = QNEW[m, lpbc]

@njit(parallel=True)
def compute_FD_step(Q, QNEW, U, LMARK, H):
    for l in prange(NMAX):
        if LMARK[l] == LMARKBULK:
            u1 = np.zeros(2, dtype=np.float64)
            u2 = np.zeros(2, dtype=np.float64)
            u3 = np.zeros(2, dtype=np.float64)
            Q_laplacian = np.zeros(2, dtype=np.float64)
            compute_Q_laplacian(l, Q, Q_laplacian)
            compute_u1(l, Q, Q_laplacian, u1, H)
            compute_u2(l, U, u2)
            compute_u3(l, U, Q, u3)
            for m in range(2):
                QNEW[m, l] = Q[m, l] + DT * u1[m] + u2[m] + u3[m]
    calcQNEW2Q(Q, QNEW, LMARK)

# --------------------------
# Simulation loop (with file output every TIME_WRITE steps)
# --------------------------
def simulation_loop(U, F, FNEW, FEQ, P, Q, QNEW, H, SIGMA, LMARK, WKONST, E):
    for t in range(TIME_STEPS):
        if t % TIME_WRITE == 0:
            print("Writing at time step:", t)
            write_velocity(t, U)
            write_orientation(t, Q)
        compute_FD_step(Q, QNEW, U, LMARK, H)
        compute_LB_step(U, F, FNEW, FEQ, P, LMARK, WKONST, E, SIGMA, Q, H)
    print("Time step:", TIME_STEPS)
    # Final write-out
    write_velocity(TIME_STEPS, U)
    write_orientation(TIME_STEPS, Q)

# --------------------------
# Output functions (remain in Python)
# --------------------------
def write_velocity(t, U):
    filename = f"./{FILE_NAME}_velocity_{t}.dat"
    with open(filename, "w") as f:
        for l in range(NMAX):
            f.write(f"{i_vr(l)} {j_vr(l)} {U[0, l]} {U[1, l]} {U[2, l]}\n")

def write_orientation(t, Q):
    filename = f"./{FILE_NAME}_orientation_{t}.dat"
    with open(filename, "w") as f:
        for l in range(NMAX):
            degree_of_order = math.sqrt(4 * (Q[0, l]**2 + Q[1, l]**2))
            angle = 0.5 * math.atan2(Q[1, l], Q[0, l])
            f.write(f"{i_vr(l)} {j_vr(l)} {degree_of_order} {angle}\n")

# --------------------------
# Main entry point
# --------------------------
def main():
    random.seed(12345)
    np.random.seed(12345)
    
    E, WKONST = initialise_E()
    
    # Allocate global arrays
    U = np.zeros((3, NMAX), dtype=np.float64)
    F = np.zeros((LATTICE_VELOCITY_NUMBER, NMAX), dtype=np.float64)
    FNEW = np.zeros((LATTICE_VELOCITY_NUMBER, NMAX), dtype=np.float64)
    FEQ = np.zeros((LATTICE_VELOCITY_NUMBER, NMAX), dtype=np.float64)
    P = np.zeros((LATTICE_VELOCITY_NUMBER, NMAX), dtype=np.float64)
    Q = np.zeros((2, NMAX), dtype=np.float64)
    QNEW = np.zeros((2, NMAX), dtype=np.float64)
    H = np.zeros((2, NMAX), dtype=np.float64)
    SIGMA = np.zeros((4, NMAX), dtype=np.float64)
    LMARK = np.zeros(NMAX, dtype=np.int64)
    
    # Define defect locations for the Q-tensor initialization
    x1 = I // 4
    y1 = J // 2
    x2 = 3 * I // 4
    y2 = J // 2
    
    # Initialize fields
    for l in range(NMAX):
        i_coord = i_vr(l)
        j_coord = j_vr(l)
        if i_coord == 0 or i_coord == I - 1 or j_coord == 0 or j_coord == J - 1:
            LMARK[l] = LMARKBC
        else:
            LMARK[l] = LMARKBULK
        
        angle = 2 * math.pi * random.random()
        U[0, l] = DENSITYINIT
        U[1, l] = 0.10  # velocity components initially zero
        U[2, l] = 0.10
        
        theta1 = 0.5 * math.atan2(j_coord - y1, i_coord - x1)
        theta2 = -0.5 * math.atan2(j_coord - y2, i_coord - x2)
        theta = theta1 + theta2
        degree_of_order = 1.0
        Q[0, l] = degree_of_order / 2.0 * math.cos(2 * theta)
        Q[1, l] = degree_of_order / 2.0 * math.sin(2 * theta)
    
    computeFeq(U, FEQ, WKONST, LMARK, E)
    for l in range(NMAX):
        for m in range(LATTICE_VELOCITY_NUMBER):
            F[m, l] = FEQ[m, l]
            FNEW[m, l] = F[m, l]
    
    # Run simulation with file output every TIME_WRITE steps
    simulation_loop(U, F, FNEW, FEQ, P, Q, QNEW, H, SIGMA, LMARK, WKONST, E)

if __name__ == '__main__':
    main()