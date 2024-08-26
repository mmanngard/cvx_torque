import cvxpy as cp
import numpy as np
import scipy.linalg as la
import time

def c2d(sys, ts):
    """
    C2D computes a discrete-time model of a system (A_c,B_c) with sample time ts.
    The function returns matrices Ad, Bd of the discrete-time system.
    """
    A = sys.A
    B = sys.B
    m, n = A.shape
    nb = B.shape[1]

    s = np.concatenate([A,B], axis=1)
    s = np.concatenate([s, np.zeros((nb, n+nb))], axis=0)
    S = la.expm(s*ts)
    Ad = S[0:n,0:n]
    Bd = S[0:n,n:n+nb+1]
    sysd = DiscreteSystem(A=Ad,B=Bd,C=sys.C,D=sys.D,ts=ts)
    return sysd

class ContinuousSystem:
    def __init__(self, A, B, C, D):
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.D = np.array(D)
        self.get_dims()

    def get_dims(self):
        a_shape = np.shape(self.A)
        b_shape = np.shape(self.B)
        c_shape = np.shape(self.C)
        self.n_states = a_shape[0]
        self.n_inputs = b_shape[1]
        self.n_outputs = c_shape[0]
        return self.n_states, self.n_inputs, self.n_outputs
    
    def obsm(self):
        pass

    def contm(self):
        pass

class DiscreteSystem:
    def __init__(self, A, B, C, D, ts):
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.D = np.array(D)
        self.ts = ts
        self.get_dims()

    def get_dims(self):
        a_shape = np.shape(self.A)
        b_shape = np.shape(self.B)
        c_shape = np.shape(self.C)
        self.n_states = a_shape[0]
        self.n_inputs = b_shape[1]
        self.n_outputs = c_shape[0]
        return self.n_states, self.n_inputs, self.n_outputs
    
    def gamma(self, n):
        '''
        Create the impulse response matrix used in the data equation.

        Parameters:

        A : numpy.ndarray
            The state matrix of the state-space system
        B : numpy.ndarray
            The input matrix of the state-space system
        C : numpy.ndarray
            The observation matrix of the state-space system
        n : float
            number of measurements

        Returns:

        gamma : numpy.ndarray, shape(n*number of state variables, n*number of state variables)
            The impulse response matrix
        '''
        A = self.A
        B = self.B
        C = self.C
        D = self.D

        A_power = np.copy(A)
        Z = np.zeros((C @ B).shape)

        # first column
        gamma_column_first = np.vstack((
            Z,
            C @ B,
            C @ A @ B
        ))
        for _ in range(n-3):
            A_power = A_power @ A
            gamma_column_first = np.vstack((gamma_column_first, C @ A_power @ B))

        # build complete matrix, column by column, from left to right
        gamma = np.copy(gamma_column_first)
        current_column = 1
        for _ in range(1, n):
            gamma_rows = Z

            # first add zero matrices
            for _ in range(current_column):
                gamma_rows = np.vstack((gamma_rows, Z))

            # then add the impulse responses
            A_power2 = np.copy(A)

            if current_column < (n-2):
                gamma_rows = np.vstack((
                    gamma_rows,
                    C @ B,
                    C @ A @ B # these must not be added to the last and the second to last columns
                ))

            if current_column == (n-2):
                gamma_rows = np.vstack((
                    gamma_rows,
                    C @ B # this has to be added to the end of the second to last column
                ))

            for _ in range(n-current_column-3):
                A_power2 = A_power2 @ A
                gamma_rows = np.vstack((gamma_rows, C @ A_power2 @ B))

            # add column on the right hand side
            gamma = np.hstack((gamma, gamma_rows))
            current_column += 1

        return gamma

    def obsm(self, n):
        '''
        Create the extended observability matrix used in the data equation.

        Parameters:

        A : numpy.ndarray
            The state matrix of the state-space system
        C : numpy.ndarray
            The observation matrix of the state-space system
        n : float
            number of measurements

        Returns:

        O : numpy.ndarray, shape(n, number of state variables)
            The extended observability matrix
        '''
        A_power = np.copy(self.A)
        O = np.vstack((np.copy(self.C), self.C @ self.A))

        for k in range(n-2):
            A_power = A_power @ self.A
            O = np.vstack((O, self.C @ A_power))

        return O

    def contm(self, n):
        pass
    
    def dataeq(self, n):
        O = self.obsm(n)
        Gamma = self.gamma(n)

        return O, Gamma

def second_difference_matrix(n, m):
    '''
    n: number of data points
    m: number of inputs to system
    '''
    D2 = np.eye(n*m) - 2*np.eye(n*m, k=2) + np.eye(n*m, k=4)
    # delete incomplete rows
    D2 = D2[:-2*m, :]

    return D2

#class VirtualSensor:
#    def __init__(self, vars, pars, method='svr'):
#        self.method = method
#        self.pars = pars
#        self.vars = vars
#
#        # construct problem on init
#        self.update_problem()
#        
#    def update_problem(self):
#        '''create the cvx problem'''
#        # define objective function
#        objective = cp.Minimize(cp.sum_squares(self.pars.D2 @ self.vars.u) + self.pars.lam*cp.norm1(self.vars.zeta + self.vars.zeta_ast))
#
#        # define constraints
#        constraints = [
#            self.pars.y - self.pars.Gamma @ self.vars.u - self.pars.O @ self.pars.x0 <= self.pars.eps + self.vars.zeta,
#            self.pars.Gamma @ self.vars.u + self.pars.O @ self.pars.x0 - self.pars.y <= self.pars.eps + self.vars.zeta_ast,
#            self.vars.zeta >= 0,
#            self.vars.zeta_ast >= 0
#        ]
#
#        self.problem = cp.Problem(objective, constraints)
#
#    def solve(self):
#        self.problem.solve()
#        print("prob solved ")
#
#    def get_plots(self):
#        pass

# create data structure
class DataClass:
    def __init__(self,t,y):
        self.t = t
        self.y = y
    def len(self):
        return np.len(self.y)

