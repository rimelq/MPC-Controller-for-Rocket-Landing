import numpy as np
import cvxpy as cp
from control import dlqr
from .MPCControl_base import MPCControl_base
from mpt4py import Polyhedron
import time
import matplotlib.pyplot as plt


def interval_vertices_from_Hrep(P: Polyhedron, tol: float = 1e-12):
    A = np.array(P.A, dtype=float).reshape(-1)
    b = np.array(P.b, dtype=float).reshape(-1)
    u_min = -np.inf
    u_max =  np.inf
    for a, bi in zip(A, b):
        if abs(a) < tol:
            continue
        val = bi / a
        if a > 0:     
            u_max = min(u_max, val)
        else:         
            u_min = max(u_min, val)

    return float(u_min), float(u_max)


def min_robust_invariant_set(A_cl: np.ndarray, W: Polyhedron, max_iter: int = 100) -> Polyhedron:
        nx = A_cl.shape[0]
        Omega = W
        itr = 0
        A_cl_ith_power = np.eye(nx)
        while itr <= max_iter:
            A_cl_ith_power = np.linalg.matrix_power(A_cl, itr)
            Omega_next = Omega + A_cl_ith_power @ W
            Omega_next.minHrep() 
            if np.linalg.matrix_norm(A_cl_ith_power, ord=2) < 5e-2:
                print('Minimal robust invariant set computation converged after {0} iterations.'.format(itr))
                break

            if itr == max_iter:
                print(np.linalg.matrix_norm(A_cl_ith_power, ord=2))
                print('Minimal robust invariant set computation did NOT converge after {0} iterations.'.format(itr))
            
            Omega = Omega_next
            itr += 1
        return Omega_next


def max_invariant_set(A_cl, X: Polyhedron, max_iter = 30) -> Polyhedron:
        O = X
        itr = 1
        converged = False
        while itr < max_iter:
            Oprev = O
            F, f = O.A, O.b
            O = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)), np.vstack((f, f)).reshape((-1,)))
            O.minHrep(True)
            _ = O.Vrep
            if O == Oprev:
                converged = True
                break
            itr += 1
        
        if converged:
            print('Maximum invariant set successfully computed after {0} iterations.'.format(itr))
        else:  
            print('Maximum invariant NOT set successfully computed after {0} iterations.'.format(itr))
        return O


class MPCControl_z(MPCControl_base):
    x_ids = np.array([8, 11])  
    u_ids = np.array([2])       

    w_min = -15.0
    w_max = 5.0

    def _setup_controller(self):

        A = np.array(self.A, dtype=float)
        B = np.array(self.B, dtype=float).reshape(2, 1)

        print("A =\n", A)
        print("B =\n", B)

        u_min, u_max = 40.0, 80.0

        self.u_lb = np.array([u_min])
        self.u_ub = np.array([u_max])

        Q = np.diag([50.0, 500.0])  
        R = np.diag([0.5])         

        K_lqr, P_lqr, _ = dlqr(A, B, Q, R)

        self.K = -np.array(K_lqr, dtype=float)  
        self.P = np.array(P_lqr, dtype=float)  
        self.A_cl = A + B @ self.K

        print("K_lqr =\n", K_lqr)
        print("K =\n", self.K)
        print("P =\n", self.P)

        print("A_cl =\n", self.A_cl)
        eigvals = np.linalg.eigvals(self.A_cl)
        print("eig(A_cl) =", eigvals)
        print("max |eig(A_cl)| =", np.max(np.abs(eigvals)))

        W = Polyhedron.from_Hrep([[1],[-1]], [5,15])   
        W = W.affine_map(B)                          

        tE = time.perf_counter()
        E = min_robust_invariant_set(self.A_cl, W)
        self.E = E

        A_X = np.array([[0.0, -1.0]])     
        b_X = np.array([self.xs[1] - 0.0])  
        X = Polyhedron.from_Hrep(A_X, b_X)

        X_tilde = X - E

        U = Polyhedron.from_Hrep(
            np.array([[1.0], [-1.0]]),
            np.array([
                u_max - self.us[0],
                self.us[0] - u_min
            ])
        )

        KE = self.K @ E

        U_tilde = U - KE
        self.U_tilde = U_tilde  
        U_tilde.minHrep()

        self.u_tilde_min, self.u_tilde_max = interval_vertices_from_Hrep(U_tilde)

        self.pavg_nom_min = float(self.us[0] + self.u_tilde_min)
        self.pavg_nom_max = float(self.us[0] + self.u_tilde_max)
  
        X_and_KU = X.intersect(Polyhedron.from_Hrep(U.A @ self.K, U.b))
        
        Xf = max_invariant_set(self.A_cl, X_and_KU)

        X_tilde_and_KU_tilde = X_tilde.intersect(Polyhedron.from_Hrep(U_tilde.A @ self.K, U_tilde.b))
        
        Xf_tilde = max_invariant_set(self.A_cl, X_tilde_and_KU_tilde)
        Xf_tilde.minHrep()
        U_tilde.minHrep()    

        self.Xf = Xf
        self.Xf_tilde = Xf_tilde

        fig3, ax3 = plt.subplots(1, 1)
        Xf.plot(ax3, color='g', opacity=0.5, label=r'$\mathcal{X}_f$')
        E.plot(ax3, color='r', opacity=0.5, label=r'$\mathcal{E}$')
        ax3.set_ylim(-4, 5)
        plt.legend()
        plt.show()

        self._x_var   = cp.Variable((2, self.N + 1))   
        self._u_var   = cp.Variable((1, self.N))       
        self._x0_p    = cp.Parameter(2)                
        self._xref_p  = cp.Parameter(2)
        self._uref_p  = cp.Parameter(1)

        constraints = []

        # tube initial condition
        constraints += [ self.E.A @ ((self._x0_p - self.xs) - self._x_var[:, 0]) <= self.E.b ]

        # dynamics on nominal
        for k in range(self.N):
            constraints += [
                self._x_var[:, k+1] == A @ self._x_var[:, k] + B.flatten() * self._u_var[:, k],
                X_tilde.A @ self._x_var[:, k] <= X_tilde.b,
                U_tilde.A @ self._u_var[:, k] <= U_tilde.b,
            ]

        # terminal
        constraints += [ self.Xf_tilde.A @ self._x_var[:, self.N] <= self.Xf_tilde.b ]

        # cost 
        cost = 0
        for k in range(self.N):
            e  = self._x_var[:, k] - self._xref_p
            du = self._u_var[:, k] - self._uref_p
            cost += cp.quad_form(e, Q) + cp.quad_form(du, R)

        cost += cp.quad_form(self._x_var[:, self.N] - self._xref_p, self.P)

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None):
        if x_target is None:
            x_target = self.xs.copy()
        if u_target is None:
            u_target = self.us.copy()

        x0 = np.asarray(x0, dtype=float).reshape(self.nx)    
        x_target = np.asarray(x_target, dtype=float).reshape(self.nx)
        u_target = np.asarray(u_target, dtype=float).reshape(self.nu)

        self._x0_p.value   = x0
        self._xref_p.value = (x_target - self.xs)            
        self._uref_p.value = (u_target - self.us)            

        self.problem.solve(warm_start=True, verbose=False, solver=cp.OSQP,
                        eps_abs=1e-5, eps_rel=1e-5, max_iter=200000, polish=True)
        

        ok = self.problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        dx = x0 - self.xs

        if (not ok) or (self._x_var.value is None) or (self._u_var.value is None):
            u0 = (self.us + (self.K @ dx).reshape(self.nu,)).copy()
            u0 = self._clip_strict(u0, self.u_lb, self.u_ub)

            x_traj = np.zeros((self.nx, self.N + 1))
            u_traj = np.zeros((self.nu, self.N))
            x_traj[:, 0] = x0
            for k in range(self.N):
                u_traj[:, k] = u0
                x_traj[:, k+1] = (
                    self.A @ (x_traj[:, k] - self.xs)
                    + self.B @ (u0 - self.us)
                    + self.xs
                )
            return u0, x_traj, u_traj

        z_traj = np.array(self._x_var.value, dtype=float)  
        v_traj = np.array(self._u_var.value, dtype=float)  

        z0 = z_traj[:, 0]
        v0 = v_traj[:, 0]

        dx = x0 - self.xs
        e0 = dx - z0

        k0 = (self.K @ e0).reshape(self.nu,)

        du_min0 = self.u_lb - self.us - v0
        du_max0 = self.u_ub - self.us - v0

        k0_sat = np.clip(k0, du_min0, du_max0)

        du0 = v0 + k0_sat
        u0  = (self.us + du0).copy()  

        x_traj = np.zeros((self.nx, self.N + 1))
        u_traj = np.zeros((self.nu, self.N))
        x_traj[:, 0] = x0

        for k in range(self.N):
            ek = (x_traj[:, k] - self.xs) - z_traj[:, k]
            vk = v_traj[:, k]

            kk = (self.K @ ek).reshape(self.nu,)

            du_mink = self.u_lb - self.us - vk
            du_maxk = self.u_ub - self.us - vk

            kk_sat = np.clip(kk, du_mink, du_maxk)

            u_traj[:, k] = self.us + vk + kk_sat

            x_traj[:, k+1] = (
                self.A @ (x_traj[:, k] - self.xs)
                + self.B @ (u_traj[:, k] - self.us)
                + self.xs
            )

        return u0, x_traj, u_traj



        