import numpy as np

class ConjugateGradient:
  def __init__(self, A, b, x0, tol = 1e-6, max_iter = 1000):
    self.A = A
    self.b = b
    self.x = x0
    self.tol = tol
    self.max_iter = max_iter

  def solve(self):
    # Initialization
    r = self.A @ self.x - self.b
    p = -r

    # Create error list to store the mean square error for all dimensions
    r_ls = [np.square(r).mean()]
    while True:

      r_sq = np.dot(r,r)

      # Calculate step size alpha
      Ap = self.A @ p
      alpha = r_sq / np.dot(p, Ap)
      # Update x
      self.x += alpha * p
      # Update r and error list
      r += alpha * Ap
      r_ls.append(np.square(r).mean())
      r_sq_new = np.dot(r,r)

      # Evaluate optimality
      if r_sq_new < self.tol:
        break

      beta = r_sq_new / r_sq
      p = -r + beta * p
      r_sq = r_sq_new

    return self.x, r_ls