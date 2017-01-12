import numpy as np
import math
import cmath
import scipy.linalg
from scipy.stats import norm
import matplotlib.pyplot as plt
from simu import *
from scipy.interpolate import interp1d



def transpo(m):
	n = len(m)
	t = []
	for i in range(n):
		c = []
		for j in range(n):
			c += [ m[j][i] ]
		t += [c]
	return t

#m = [ [1, 2, 3], [4, 5, 6], [7, 8, 9] ]
#print(m)
#print(transpo(m))




CAPTVALUE = 8
SAMPLVALUE = 1000
MAKEREF = 10
DAMAGED_SENSOR = [5]#must be a list
DAMAGE = 70


pos = [ k for k in range(CAPTVALUE) ]
mass = [ 100 for k in range(CAPTVALUE) ]
stiffness = [ 100 for k in range(CAPTVALUE) ]


N = SAMPLVALUE
n = len(mass)
M = matrix_M(mass)
K = matrix_K(stiffness)

#print( M )
#print( K )

Minv = np.linalg.inv(M)
MinvK = np.dot(Minv,K)

eigenvalues, ev = np.linalg.eig(MinvK)


frq = []
for i in eigenvalues:
		frq += [math.sqrt(abs(i))/(2*math.pi)]

frq = frq[::-1]
#print(frq)

ev = ev[::-1]
ev = transpo(ev)


for x in DAMAGED_SENSOR:
	stiffness[x] = DAMAGE

N = SAMPLVALUE
n = len(mass)
M = matrix_M(mass)
K = matrix_K(stiffness)
C = matrix_C(M,K,0.005,0.005)


Minv = np.linalg.inv(M)
MinvK = np.dot(Minv,K)

eigenvalues, ev_damaged = np.linalg.eig(MinvK)

ev_damaged = ev_damaged[::-1]
ev_damaged = transpo(ev_damaged)
"""
for k in range(len(ev)):
	plt.plot(pos,ev[k])
	plt.plot(pos,ev_damaged[k])
	plt.title("Vecteur " + str(k))
	plt.show()
"""

error = [ 0 for k in range(len(ev)) ]
error_damaged = [ 0 for k in range(len(ev)) ]

for k in range(len(ev)):
	for i in range(len(ev[k]))[1:-1]:
		x = []
		y = []
		yd = []
		for j in range(len(ev[k])):
			if (j != i):
				x += [pos[j]]
				y += [ev[k][j]]
				yd += [ev_damaged[k][j]]
		#print(y)
		f = interp1d(x, y, kind = 'cubic')
		g = interp1d(x, yd, kind = 'cubic')
		v_int = f( pos[i] )
		v_int_damaged = g(pos[i])
		y = []
		yd = []
		for j in range(len(ev[k])):
			if (j != i):
				y += [ev[k][j]]
				yd += [ev_damaged[k][j]]
			else:
				y += [v_int]
				yd += [v_int_damaged]
		error[i] += abs(v_int - ev[k][i])
		error_damaged[i] += abs(v_int_damaged - ev[k][i])
		#plt.subplot(121)
		#plt.plot(pos,ev[k])
		#plt.plot(pos,y)
		#plt.subplot(122)
		#plt.plot(pos,ev_damaged[k])
		#plt.plot(pos,yd)
		#plt.show()

diff = error_damaged
for k in range(len(diff)):
	diff[k] -= error[k]

print("Damage on: " + str(DAMAGED_SENSOR))
#print("Undamaged:")
#print(error)
#print("Damaged:")
#print(error_damaged)
print("Diff:")
print(diff)

