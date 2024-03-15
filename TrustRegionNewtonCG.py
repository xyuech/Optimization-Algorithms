import numpy as np


class TrustRegionNewton:
    def __init__(self, objective_fn, gradient_fn, hessian_fn, max_iter=1000, tol=1e-6, eta=0.25, beta=0.25):
        '''
        Class Attribute:
          objective_fn: minimization objective function
          gradient_fn: explicit gradient function
          hessian_fn: B_k = the hessian matrix, newton method
          eta: expansion threshold for radius
          beta: reduction ratio for radius
          tol: the convergence threshold
        Description:
          minimize the function f(x)
          reformulate by taylor's expansion as m(p)=f(x+p)=f(x) + g(x)^T + 1/2 p^TH(x)p w.r.t descent steps p
          and minize m(p) over p, subject to |p| <= TR radius

          Trust region:
          - Find p minimizing m(p)
          - Evaluate change of ratio rho
            - Change radius according to rho
            - Update x = x + p
            - Terminate if g(x) <= tol

          Method to find p
          - Conjugate gradient method: solve Bp = g and satisfy |p| <= TR radius
          - Cauchy point method
          - Dogleg method
          - Iterative method
        '''
        self.objective_fn = objective_fn
        self.gradient_fn = gradient_fn
        self.hessian_fn = hessian_fn
        self.max_iter = max_iter  # max iteration for k (Trust Region)
        self.tol = tol  # convergence threshold
        self.eta = eta  # trust radius
        self.beta = beta  # shrinkage factor for radius

    # @staticmethod
    def cg_solve(self, hessian, gradient, trust_radius):
        '''
        Description:
          CG used to find minimizer p of m(p) subject to |p| < TR radius
          Equivalent to find solution for Bp = g
        Input:
          hessian: hessian matrix (B)
          gradient: gradient function
          trust_radius: initial trust region radius
        Output:
          p: descent direction: feasible solution for Bp = g, satisfying constraints, p = z + tau * d
        '''
        # Initialization for Conjugate Gradient
        
        n = len(gradient)
        z = np.zeros_like(gradient) # initial z (p in m(p))
        r = gradient.copy()  # current descent steps (direction and step length included)
        d = -r  # conjugate direction for B
        j = 0  # iteration counter

        if np.linalg.norm(r) < self.tol:
            # print("   Case 0: Small initial residual.")
            # print(f"    Conjugate gradient take iterations: ", j)
            return z, j

        while True:
            # Check optimality

            j += 1
            # Bd
            Bd = hessian @ d
            # dBd
            dBd = d.transpose() @ Bd

            if dBd <= 0:
                # Negative curvature encountered, m(z_new) = m(z + tau * d) decrease with tau
                zd = z.transpose() @ d; dd = d.transpose() @ d; zz = z.transpose() @ z
                # Caculate max tau that makes p = Delta
                tau = (-zd + np.sqrt(zd**2 - dd * (zz - trust_radius**2))) / dd
                z += tau * d
                # print("    Case 1: Negative curvature.")
                # print(f"    Conjugate gradient take iterations: ", j)
                return z, j

            # Conjugate Gradient

            # Calculate step-size
            alpha = (r.transpose() @ r) / dBd
            # Update z
            z_new = z + alpha * d

            if np.linalg.norm(z_new) >= trust_radius:
                # Trust region bound reached, stopped at boundary
                zd = z.transpose() @ d; dd = d.transpose() @ d; zz = z.transpose() @ z
                # Caculate max tau that makes p = Delta
                tau = (-zd + np.sqrt(zd**2 - dd * (zz - trust_radius**2))) / dd
                z += tau * d
                
                # print("   Case 2: Trust region bound reached.")
                # print(f"    Conjugate gradient take iterations: ", j)
                return z, j
            
            r_new = r + alpha * Bd
            # print("r_new is: ", r_new)
            if np.linalg.norm(r_new) < 1e-8:
                # print("    Case 3: Residual small enough.")
                # print(f"    Conjugate gradient take iterations: ", j)
                return z_new, j 

            beta = (r_new.transpose() @ r_new) / (r.transpose() @ r)
            d = - r_new + beta * d
            z = z_new
            r = r_new


    
    # def cauchy_solve()


    # def dogleg_solve()

    # def iter_solve()

    def minimize(self, x0, trust_radius=1.0, max_radius=10.0):
        # Choose solution method
        # solve_method = getattr(TrustRegionNewton, solve_ag)

        # Initialization
        x = x0
        j_list = [] # record CG iterations
        p_list = [] # record descent direction solved by CG

        f_list = [] # record function value at xk
        g_list = [] # record gradient function value at xk
        m_list = [] # record m function value at pk

        # rho_list = [] # record change ratio
        # rad_list = [] # record trust region radius ratio
        x_list = [x]

        for k in range(self.max_iter):
            # print(f"========== Start Iteration {k} ==========")
            gradient = self.gradient_fn(x)

            hessian = self.hessian_fn(x)
            g_list.append(np.linalg.norm(gradient))
            # print(f"gradient for iteration {k}: ", gradient)

            m = lambda p: self.objective_fn(x) + (gradient.transpose() @ p) + 0.5 * p.transpose() @ (hessian @ p)

            p, j = self.cg_solve(hessian, gradient, trust_radius)
            # print(f"p for iteration {k}: ", p) 
            # print(f"|p| for iteration {k}: ", np.linalg.norm(p)) 
            p_list.append(p)
            j_list.append(j)

            # Calculate change ratio
            fx = self.objective_fn(x)
            fx_new = self.objective_fn(x + p)
            mp = m(p)
            m0 = m(np.zeros_like(x))
            rho = (fx - fx_new) / (m0 - mp)

            # print(f"fx for iteration {k}: ", fx) 
            # print(f"fx_new for iteration {k}: ", fx_new) 
            # print(f"m(0) for iteration {k}: ", m0) 
            # print(f"m(p) for iteration {k}: ", mp) 
            # print(f"rho for iteration {k}: ", rho) 
            f_list.append(fx)
            m_list.append(mp)
            #rho_list.append[rho]
            #rad_list.append[trust_radius]

            # Change of trust region radius
            if rho < 0.25:
                trust_radius *= self.beta
                # change in f is too small w.r.t. change in m, radius too loose -> shrink the radius

            elif rho > 0.75 and np.linalg.norm(p) >= trust_radius - self.tol:
                trust_radius = min(2 * trust_radius, max_radius)
                # change in f is relatively big w.r.t. change in m
                # and the length current step p is equal to radius
                # meaning that current direction decreases f(x) and hasn't reach the minimum
                # -> increase the radius

            if rho > self.eta:
                x_new = x + p
            
            # print(f"trust_radius for iteration {k}: ", trust_radius)   

            # Termination condition depends on the function value
            if np.linalg.norm(fx_new - fx) < self.tol:
            #if np.linalg.norm(gradient) < self.tol:
                break
            x = x_new
            x_list.append(x)

        return x_list, f_list, g_list, m_list, p_list, j_list, k


