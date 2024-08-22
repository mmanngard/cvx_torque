import virtual_sensors as vs
import numpy as np

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
data = vs.DataClass(t=np.array([[0],[1],[2]]), y=np.array([[1],[-1], [2]]))

pars = {
    'eps': 0.1,
    'lam': 1,
    'x0': np.zeros((2,1))
    }

# create the problem
virtual_sensor = vs.VirtualSensor(sysd, data, pars, method='svr')

#solve the problem
virtual_sensor.solve()

#print result
print(virtual_sensor.u.value)