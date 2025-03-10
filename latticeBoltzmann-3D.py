"""
Author: Ashesh Ghosh
Last Update: Feb 12 2025 (extended Feb 24 2025)
Description: Optimized simulation code for active nematic in 3d using the lattice Boltzmann method.
"""

import numpy as np
import math
import random
from numba import njit, prange # type: ignore

# --------------------------
# Parameters
# --------------------------
FILE_NAME = "active_nematic_3d"

# Lattice dimensions
I = 50       # x-direction
J = 50       # y-direction
K = 50        # z-direction
NMAX = I * J * K

TIME_STEPS = 1000
TIME_WRITE = 100

# Lattice Boltzmann parameters for D3Q19
LATTICE_VELOCITY_NUMBER = 19
STPROC = 2  # (parallelism handled by Numba)

C = 1.0
DT = 0.01
TAUF = 2.0 * DT
DENSITYINIT = 2.0 / DT
LAMBDA = 1.1
ALPHA = 1.00

LMARKBULK = 2    # bulk
LMARKBC = 4      # boundary condition

# For Q-tensor (5 independent components in 3d)
# We will represent Q as [Q_xx, Q_xy, Q_xz, Q_yy, Q_yz]
# with Q_zz = -Q_xx - Q_yy
# (a uniaxial state along x can be initialized with S=1, for example)

# --------------------------
# 3d Indexing functions
# --------------------------
@njit(inline='always')
def i_vr(l):
    return l % I

@njit(inline='always')
def j_vr(l):
    return (l // I) % J

@njit(inline='always')
def k_vr(l):
    return l // (I * J)

@njit(inline='always')
def index_from_coords(i, j, k):
    return i + I * (j + J * k)

# --------------------------
# Initialization of D3Q19 lattice vectors and weights
# --------------------------
@njit
def initialise_E():
    # D3Q19 discrete velocities and weights
    E = np.zeros((LATTICE_VELOCITY_NUMBER, 3), dtype=np.int64)
    WKONST = np.zeros(LATTICE_VELOCITY_NUMBER, dtype=np.float64)
    # Zero velocity
    E[0, 0] = 0; E[0, 1] = 0; E[0, 2] = 0
    WKONST[0] = 1.0/3.0

    # 6 face–centered directions (±x, ±y, ±z)
    E[1, :] = (1, 0, 0)
    E[2, :] = (-1, 0, 0)
    E[3, :] = (0, 1, 0)
    E[4, :] = (0, -1, 0)
    E[5, :] = (0, 0, 1)
    E[6, :] = (0, 0, -1)
    for i in range(1, 7):
        WKONST[i] = 1.0/18.0

    # 12 edge–centered directions
    E[7, :]  = (1, 1, 0)
    E[8, :]  = (-1, -1, 0)
    E[9, :]  = (1, -1, 0)
    E[10, :] = (-1, 1, 0)
    E[11, :] = (1, 0, 1)
    E[12, :] = (-1, 0, -1)
    E[13, :] = (1, 0, -1)
    E[14, :] = (-1, 0, 1)
    E[15, :] = (0, 1, 1)
    E[16, :] = (0, -1, -1)
    E[17, :] = (0, 1, -1)
    E[18, :] = (0, -1, 1)
    for i in range(7, 19):
        WKONST[i] = 1.0/36.0

    return E, WKONST

# --------------------------
# LB helper functions for 3d streaming and boundary conditions
# --------------------------
@njit(inline='always')
def calcLBlnew(l, m, E):
    # get current coordinates and add the lattice vector
    i_coord = i_vr(l) + E[m, 0]
    j_coord = j_vr(l) + E[m, 1]
    k_coord = k_vr(l) + E[m, 2]
    return index_from_coords(i_coord, j_coord, k_coord)

@njit(inline='always')
def calcLpbc(l):
    # Apply periodic boundary corrections in all three directions
    xp2 = 0
    yp2 = 0
    zp2 = 0
    i_coord = i_vr(l)
    j_coord = j_vr(l)
    k_coord = k_vr(l)
    if i_coord == 0:
        xp2 = I - 2
    if i_coord == I - 1:
        xp2 = -(I - 2)
    if j_coord == 0:
        yp2 = I * (J - 2)
    if j_coord == J - 1:
        yp2 = -I * (J - 2)
    if k_coord == 0:
        zp2 = I * J * (K - 2)
    if k_coord == K - 1:
        zp2 = -I * J * (K - 2)
    return l + xp2 + yp2 + zp2

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
        fez = 0.0
        for m in range(LATTICE_VELOCITY_NUMBER):
            density += F[m, l]
            fex += F[m, l] * E[m, 0]
            fey += F[m, l] * E[m, 1]
            fez += F[m, l] * E[m, 2]
        U[1, l] = fex / density
        U[2, l] = fey / density
        U[3, l] = fez / density
        U[0, l] = density
    else:
        U[0, l] = DENSITYINIT
        U[1, l] = 0.0
        U[2, l] = 0.0
        U[3, l] = 0.0

# --------------------------
# LB functions in 3d
# --------------------------
@njit(parallel=True)
def computeFeq(U, FEQ, WKONST, LMARK, E):
    for l in prange(NMAX):
        if LMARK[l] != LMARKBULK:
            continue
        # three velocity components now
        u2 = U[1, l]**2 + U[2, l]**2 + U[3, l]**2
        for m in range(LATTICE_VELOCITY_NUMBER):
            ue = U[1, l] * E[m, 0] + U[2, l] * E[m, 1] + U[3, l] * E[m, 2]
            FEQ[m, l] = WKONST[m] * U[0, l] * (1.0 + 3.0 * ue + 4.5 * ue * ue - 1.5 * u2)

# In 3d, we need to compute the forcing term from the divergence of the stress tensor.
# Here we assume SIGMA is stored as a 9-component array (row–major, 3×3 tensor) per lattice node.
@njit(parallel=True)
def compute_sigma(SIGMA, H, Q, LMARK):
    # For simplicity, we define the following (naive) extension.
    # Q now has 5 components; we simply extend the 2d formulas to the xx and xy-like components.
    # Here we set:
    #   SIGMA_xx = -LAMBDA * H[0] + ALPHA * Q[0]
    #   SIGMA_xy = -LAMBDA * H[1] + ALPHA * Q[1]
    #   SIGMA_xz = -LAMBDA * H[2] + ALPHA * Q[2]
    #   SIGMA_yx = LAMBDA * H[0] - ALPHA * Q[0]
    #   SIGMA_yy = -LAMBDA * H[3] + ALPHA * Q[3]
    #   SIGMA_yz = -LAMBDA * H[4] + ALPHA * Q[4]
    #   SIGMA_zx = SIGMA_xy
    #   SIGMA_zy = SIGMA_yz
    #   SIGMA_zz = LAMBDA * H[3] - ALPHA * Q[3]  # (as a simple closure)
    for l in prange(NMAX):
        if LMARK[l] != LMARKBULK:
            continue
        SIGMA[0, l] = -LAMBDA * H[0, l] + ALPHA * Q[0, l]  # sigma_xx
        SIGMA[1, l] = -LAMBDA * H[1, l] + ALPHA * Q[1, l]  # sigma_xy
        SIGMA[2, l] = -LAMBDA * H[2, l] + ALPHA * Q[2, l]  # sigma_xz
        SIGMA[3, l] = LAMBDA * H[0, l] - ALPHA * Q[0, l]   # sigma_yx
        SIGMA[4, l] = -LAMBDA * H[3, l] + ALPHA * Q[3, l]  # sigma_yy
        SIGMA[5, l] = -LAMBDA * H[4, l] + ALPHA * Q[4, l]  # sigma_yz
        SIGMA[6, l] = SIGMA[1, l]                           # sigma_zx
        SIGMA[7, l] = SIGMA[5, l]                           # sigma_zy
        SIGMA[8, l] = LAMBDA * H[3, l] - ALPHA * Q[3, l]    # sigma_zz (naively)
    # Apply periodic BC for sigma:
    for l in prange(NMAX):
        lpbc = calcLpbc(l)
        if lpbc != l:
            for m in range(9):
                SIGMA[m, l] = SIGMA[m, lpbc]

@njit(parallel=True)
def computeP(P, SIGMA, U, WKONST, LMARK, E, Q, H):
    compute_sigma(SIGMA, H, Q, LMARK)
    for l in prange(NMAX):
        if LMARK[l] != LMARKBULK:
            continue
        # Compute force components from central differences of sigma.
        # Here we assume the ordering in SIGMA:
        # [0]=xx, [1]=xy, [2]=xz, [3]=yx, [4]=yy, [5]=yz, [6]=zx, [7]=zy, [8]=zz.
        # Force_x = d/dx sigma_xx + d/dy sigma_xy + d/dz sigma_xz.
        # Use central differences in each direction.
        lxp = l + 1
        lxm = l - 1
        lyp = l + I
        lym = l - I
        lzp = l + I*J
        lzm = l - I*J
        forceX = ((SIGMA[0, lxp] - SIGMA[0, lxm]) / 2.0 +
                  (SIGMA[1, lyp] - SIGMA[1, lym]) / 2.0 +
                  (SIGMA[2, lzp] - SIGMA[2, lzm]) / 2.0)
        forceY = ((SIGMA[3, lxp] - SIGMA[3, lxm]) / 2.0 +
                  (SIGMA[4, lyp] - SIGMA[4, lym]) / 2.0 +
                  (SIGMA[5, lzp] - SIGMA[5, lzm]) / 2.0)
        forceZ = ((SIGMA[6, lxp] - SIGMA[6, lxm]) / 2.0 +
                  (SIGMA[7, lyp] - SIGMA[7, lym]) / 2.0 +
                  (SIGMA[8, lzp] - SIGMA[8, lzm]) / 2.0)
        uF = U[1, l] * forceX + U[2, l] * forceY + U[3, l] * forceZ
        for m in range(LATTICE_VELOCITY_NUMBER):
            ue = (U[1, l] * E[m, 0] + U[2, l] * E[m, 1] + U[3, l] * E[m, 2])
            eF = (E[m, 0] * forceX + E[m, 1] * forceY + E[m, 2] * forceZ)
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
            # Skip if out of range (periodic BC will fix it)
            if lnew >= NMAX or lnew < 0:
                continue
            FNEW[m, lnew] = F[m, l] - DT / TAUF * (F[m, l] - FEQ[m, l]) + P[m, l]
    calcFNEW2F(F, FNEW, LMARK)
    for l in prange(NMAX):
        calcF2U(l, U, F, E, LMARK)
    for l in prange(NMAX):
        lpbc = calcLpbc(l)
        # Ensure macroscopic variables satisfy BC
        for m in range(4):
            U[m, l] = U[m, lpbc]

# --------------------------
# Finite Difference functions for Q-tensor evolution in 3d
# --------------------------
@njit(inline='always')
def compute_Q_laplacian(l, Q, Q_laplacian):
    # Q has 5 components; use 6 nearest neighbors in 3d
    lxp = l + 1
    lxm = l - 1
    lyp = l + I
    lym = l - I
    lzp = l + I*J
    lzm = l - I*J
    for m in range(5):
        Q_laplacian[m] = (Q[m, lxp] + Q[m, lxm] +
                          Q[m, lyp] + Q[m, lym] +
                          Q[m, lzp] + Q[m, lzm] - 6 * Q[m, l])

@njit(inline='always')
def compute_u1(l, Q, Q_laplacian, u1, H):
    # Compute the local invariant: here we use a naive extension.
    # For 3d, one can show that Q:Q = 2*(Q_xx^2 + Q_xy^2 + Q_xz^2 + Q_yy^2 + Q_yz^2 +
    #                  Q_xx*Q_yy)  (using Q_zz = -Q_xx - Q_yy).
    inv = (2 * (Q[0, l]**2 + Q[1, l]**2 + Q[2, l]**2 + Q[3, l]**2 + Q[4, l]**2)
           + 2 * Q[0, l] * Q[3, l])
    for m in range(5):
        u1[m] = Q_laplacian[m] - 2 * C * Q[m, l] * (inv - 1.0)
        H[m, l] = u1[m]

@njit(inline='always')
def compute_u2(l, U, u2):
    # A naive flow-alignment term.
    # Compute a central difference of the x-component of velocity as a proxy.
    # (A proper 3d treatment would use the full symmetric velocity gradient.)
    uxx = (U[1, l+1] - U[1, l-1]) / 2.0
    # Here we use the average of y and z gradients
    uyy = (U[2, l+I] - U[2, l-I]) / 2.0
    uzz = (U[3, l+I*J] - U[3, l-I*J]) / 2.0
    avg_grad = (uxx + uyy + uzz) / 3.0
    for m in range(5):
        u2[m] = LAMBDA * avg_grad

@njit(inline='always')
def compute_u3(l, U, Q, u3):
    # Compute advection term: -(U · grad) Q for each component.
    dQ_dx = np.empty(5, dtype=np.float64)
    dQ_dy = np.empty(5, dtype=np.float64)
    dQ_dz = np.empty(5, dtype=np.float64)
    for m in range(5):
        dQ_dx[m] = (Q[m, l+1] - Q[m, l-1]) / 2.0
        dQ_dy[m] = (Q[m, l+I] - Q[m, l-I]) / 2.0
        dQ_dz[m] = (Q[m, l+I*J] - Q[m, l-I*J]) / 2.0
        u3[m] = -(U[1, l] * dQ_dx[m] + U[2, l] * dQ_dy[m] + U[3, l] * dQ_dz[m])

@njit(parallel=True)
def calcQNEW2Q(Q, QNEW, LMARK):
    for l in prange(NMAX):
        lpbc = calcLpbc(l)
        for m in range(5):
            Q[m, l] = QNEW[m, lpbc]

@njit(parallel=True)
def compute_FD_step(Q, QNEW, U, LMARK, H):
    for l in prange(NMAX):
        if LMARK[l] == LMARKBULK:
            u1 = np.zeros(5, dtype=np.float64)
            u2 = np.zeros(5, dtype=np.float64)
            u3 = np.zeros(5, dtype=np.float64)
            Q_laplacian = np.zeros(5, dtype=np.float64)
            compute_Q_laplacian(l, Q, Q_laplacian)
            compute_u1(l, Q, Q_laplacian, u1, H)
            compute_u2(l, U, u2)
            compute_u3(l, U, Q, u3)
            for m in range(5):
                QNEW[m, l] = Q[m, l] + DT * (u1[m] + u2[m] + u3[m])
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
            f.write(f"{i_vr(l)} {j_vr(l)} {k_vr(l)} {U[0, l]} {U[1, l]} {U[2, l]} {U[3, l]}\n")

def write_orientation(t, Q):
    filename = f"./{FILE_NAME}_orientation_{t}.dat"
    with open(filename, "w") as f:
        for l in range(NMAX):
            # For 3d Q-tensor one might output the five independent components.
            # Alternatively, one can compute the scalar order parameter S and the director.
            # Here we simply write out the five Q-components.
            f.write(f"{i_vr(l)} {j_vr(l)} {k_vr(l)} {Q[0, l]} {Q[1, l]} {Q[2, l]} {Q[3, l]} {Q[4, l]}\n")

# --------------------------
# Main entry point
# --------------------------
def main():
    random.seed(12345)
    np.random.seed(12345)
    
    E, WKONST = initialise_E()
    
    # Allocate global arrays
    # U now has 4 components: density, Ux, Uy, Uz.
    U = np.zeros((4, NMAX), dtype=np.float64)
    F = np.zeros((LATTICE_VELOCITY_NUMBER, NMAX), dtype=np.float64)
    FNEW = np.zeros((LATTICE_VELOCITY_NUMBER, NMAX), dtype=np.float64)
    FEQ = np.zeros((LATTICE_VELOCITY_NUMBER, NMAX), dtype=np.float64)
    P = np.zeros((LATTICE_VELOCITY_NUMBER, NMAX), dtype=np.float64)
    # Q-tensor with 5 independent components
    Q = np.zeros((5, NMAX), dtype=np.float64)
    QNEW = np.zeros((5, NMAX), dtype=np.float64)
    # H has the same shape as Q
    H = np.zeros((5, NMAX), dtype=np.float64)
    # SIGMA: 9 components per node for the stress tensor in 3d.
    SIGMA = np.zeros((9, NMAX), dtype=np.float64)
    LMARK = np.zeros(NMAX, dtype=np.int64)
    
    # Initialize fields: use periodic boundaries in the bulk and set BC markers on the boundaries.
    for l in range(NMAX):
        i_coord = i_vr(l)
        j_coord = j_vr(l)
        k_coord = k_vr(l)
        if i_coord == 0 or i_coord == I - 1 or j_coord == 0 or j_coord == J - 1 or k_coord == 0 or k_coord == K - 1:
            LMARK[l] = LMARKBC
        else:
            LMARK[l] = LMARKBULK
        
        # Initialize velocity: density and (small) velocity components.
        U[0, l] = DENSITYINIT
        U[1, l] = 0.10
        U[2, l] = 0.10
        U[3, l] = 0.10
        
        # Initialize Q-tensor to a uniaxial state along x:
        # Let S = 1.0; then
        # Q_xx = S*(1 - 1/3) = 2/3, Q_xy = 0, Q_xz = 0, Q_yy = -1/3, Q_yz = 0.
        S = 1.0
        Q[0, l] = (2.0/3.0) * S
        Q[1, l] = 0.0
        Q[2, l] = 0.0
        Q[3, l] = -1.0/3.0 * S
        Q[4, l] = 0.0
    
    computeFeq(U, FEQ, WKONST, LMARK, E)
    for l in range(NMAX):
        for m in range(LATTICE_VELOCITY_NUMBER):
            F[m, l] = FEQ[m, l]
            FNEW[m, l] = F[m, l]
    
    # Run simulation with file output every TIME_WRITE steps
    simulation_loop(U, F, FNEW, FEQ, P, Q, QNEW, H, SIGMA, LMARK, WKONST, E)

if __name__ == '__main__':
    main()