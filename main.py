import virtual_sensors as vs
import numpy as np
import cvxpy as cp

# simple example for testing the virtual_sensors_cvx package

#system
A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Sampling time
dt = 0.1

#create system
sysc = vs.ContinuousSystem(A,B,C,D)

#convert to discrete time
sysd = vs.c2d(sys=sysc,ts=1)
t = np.array([[0],[1],[2]])
y = np.array([[1],[-1], [2]])
#data = vs.DataClass(t=np.array([[0],[1],[2]]), y=np.array([[1],[-1], [2]]))

############################################################
#calculate parameters
n_samples = len(t)
n_states, n_inputs, n_outputs = sysd.get_dims()

class Parameters:
    '''A parameter class for storing parameter values used to solve the cvx problem'''
    def __init__(self,y,x0,O,Gamma,D2,lam,eps):
        self.y = y
        self.x0 = x0
        self.O = O
        self.Gamma = Gamma
        self.D2 = D2
        self.lam = lam
        self.eps = eps

class Variables:
    '''A cvx cariable class for defining optimization variables'''
    def __init__(self,u,zeta,zeta_ast):
        self.u = u
        self.zeta = zeta
        self.zeta_ast = zeta_ast

# create parameters
Gamma = sysd.gamma(n_samples)
O = sysd.obsm(n_samples)
D2 = vs.second_difference_matrix(n_samples, n_inputs)
y = np.array([[1],[-1], [2]])
x0 = np.zeros((2,1))
lam = 0.01
eps = 0.01
pars = Parameters(y,x0,O,Gamma,D2,lam,eps)

# define variables
u = cp.Variable((n_inputs*n_samples,1))
zeta = cp.Variable((n_samples*n_outputs,1))
zeta_ast = cp.Variable((n_samples*n_outputs,1))
vars = Variables(u,zeta,zeta_ast)

# create the problem
virtual_sensor = vs.VirtualSensor(vars, pars, method='svr')

print(virtual_sensor.pars.y)

#solve the problem
virtual_sensor.solve()
