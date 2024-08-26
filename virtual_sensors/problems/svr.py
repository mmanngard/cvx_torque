import cvxpy as cp

class Parameters:
    """
    A class to store parameter values used in the convex optimization problem.
    
    Attributes:
    -----------
    y : array-like
        The target output vector.
    x0 : array-like
        The initial condition or state vector.
    O : array-like
        The output matrix in the state-space model.
    Gamma : array-like
        The input matrix in the state-space model.
    D2 : array-like
        A matrix used in the objective function for regularization.
    lam : float
        Regularization parameter for the L1 norm in the objective function.
    eps : float
        Tolerance level used in the constraints.
    """
    def __init__(self, y, x0, O, Gamma, D2, lam, eps):
        self.y = y
        self.x0 = x0
        self.O = O
        self.Gamma = Gamma
        self.D2 = D2
        self.lam = lam
        self.eps = eps

class Variables:
    """
    A class to define and store optimization variables for the convex problem.
    
    Attributes:
    -----------
    u : cvxpy.Variable
        The control input variable to be optimized.
    zeta : cvxpy.Variable
        A slack variable used in the inequality constraints.
    zeta_ast : cvxpy.Variable
        Another slack variable used in the inequality constraints.
    """
    def __init__(self, BATCH_SIZE, n_inputs, n_outputs):
        self.u = cp.Variable((n_inputs*BATCH_SIZE,1))
        self.zeta = cp.Variable((BATCH_SIZE*n_outputs,1))
        self.zeta_ast = cp.Variable((BATCH_SIZE*n_outputs,1))

class Problem:
    """
    A class that formulates and solves a convex optimization problem using the given variables and parameters.
    
    Attributes:
    -----------
    vars : Variables
        An instance of the Variables class containing the optimization variables.
    pars : Parameters
        An instance of the Parameters class containing the problem's parameters.
    problem : cvxpy.Problem
        The formulated convex optimization problem, ready to be solved.

    Methods:
    --------
    create_problem():
        Formulates the optimization problem by defining the objective and constraints.
    create_objective():
        Defines the objective function for the optimization problem.
    create_constraints():
        Defines the constraints for the optimization problem.
    update():
        Updates (re-creates) the problem instance if parameters or variables change.
    """
    def __init__(self, vars: Variables, pars: Parameters):
        self.vars = vars
        self.pars = pars
        self.problem = self.create_problem()

    def create_problem(self):
        """
        Creates the cvx convex optimization problem.
        """
        objective = self.create_objective()
        constraints = self.create_constraints()
        return cp.Problem(objective, constraints)
    
    def create_objective(self):
        """
        Defines the objective function for the convex optimization problem.
        """
        return cp.Minimize(cp.sum_squares(self.pars.D2 @ self.vars.u) + 
                           self.pars.lam * cp.norm1(self.vars.zeta + self.vars.zeta_ast))
    
    def create_constraints(self):
        """
        Defines the constraints for the convex optimization problem.
        """
        return [
            self.pars.y - self.pars.Gamma @ self.vars.u - self.pars.O @ self.pars.x0 <= self.pars.eps + self.vars.zeta,
            self.pars.Gamma @ self.vars.u + self.pars.O @ self.pars.x0 - self.pars.y <= self.pars.eps + self.vars.zeta_ast,
            self.vars.zeta >= 0,
            self.vars.zeta_ast >= 0
        ]
    
    def update(self):
        """
        Re-creates the problem instance by updating the objective and constraints.
        
        This method should be called if there are any changes to the parameters or variables
        after the problem has been initially created.
        """
        self.problem = self.create_problem()
    
    def solve(self):
        self.problem.solve()
        print("prob solved ")

