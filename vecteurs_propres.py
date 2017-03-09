import numpy as np
import math
import cmath
import scipy.linalg
from scipy.stats import norm
import matplotlib.pyplot as plt
from simu import *
from scipy.interpolate import interp1d
import random
import curve_study as cs

####################################################################################################################
############################################## Parameters ##########################################################
####################################################################################################################

CAPTVALUE = 20
DAMAGED_SENSOR = [2]#must be a list
DAMAGE = 0.20


pos = [ k for k in range(CAPTVALUE) ]
mass = [ 100 for k in range(CAPTVALUE) ]
stiffness = [ 100 for k in range(CAPTVALUE) ]
#mass = [ 100*random.randint(1,5) for k in range(CAPTVALUE) ]
#stiffness = [ 100*random.randint(1,5) for k in range(CAPTVALUE) ]


####################################################################################################################
############################################## Functions ###########################################################
####################################################################################################################


def matrix_K(stiffness):#make the stiffness matrix
	n = len(stiffness)
	K = np.zeros((n,n))
	for i in range(n-1):
		K[i][i] += stiffness[i+1]
		K[i+1][i+1] += stiffness[i+1]
		K[i+1][i] = - stiffness[i+1]
		K[i][i+1] = - stiffness[i+1]
	K[0][0] += stiffness[0]
	K[n-1][n-1] += stiffness[n-1]
	return K

def transpo(m):
	n = len(m)
	t = []
	for i in range(n):
		c = []
		for j in range(n):
			c += [ m[j][i] ]
		t += [c]
	return t

def minus_list(l,m):
	n = len(l)
	r = [ (l[k] - m[k]) for k in range(n)]
	return r		

####################################################################################################################
############################################## Undamaged state #####################################################
####################################################################################################################

M = matrix_M(mass)
K = matrix_K(stiffness)
Minv = np.linalg.inv(M)
MinvK = np.dot(Minv,K)

#print("Without damage:")
#print(stiffness)
#print( M )
#print( K )
#print(MinvK)

eigenvalues, ev = np.linalg.eig(MinvK)
#print(eigenvalues)
#frq = []
#for i in eigenvalues:
#		frq += [math.sqrt(abs(i))/(2*math.pi)]
#frq = frq[::-1]
#print(frq)
#ev = ev[::-1]

ev = transpo(ev)


####################################################################################################################
############################################## Damaged state #######################################################
####################################################################################################################

for x in DAMAGED_SENSOR:
	stiffness[x] = int(stiffness[x]*(1-DAMAGE))

M = matrix_M(mass)
K = matrix_K(stiffness)
Minv = np.linalg.inv(M)
MinvK = np.dot(Minv,K)

#print("With damage:")
#print(stiffness)
#print( K )
#print(MinvK)
eigenvalues, ev_damaged = np.linalg.eig(MinvK)
#print(eigenvalues)
#ev_damaged = ev_damaged[::-1]
ev_damaged = transpo(ev_damaged)


####################################################################################################################
############################################## Damage research #####################################################
####################################################################################################################


for k in range(len(ev)):
	if (ev[k][0] < 0):
		for j in range(len(ev[k])):
			ev[k][j] = - ev[k][j]
	if (ev_damaged[k][0] < 0):
		for j in range(len(ev_damaged[k])):
			ev_damaged[k][j] = - ev_damaged[k][j]


"""
for k in range(len(ev)):
	if (1):
		plt.plot(pos,ev[k])
		plt.plot(pos,ev_damaged[k])
		plt.title("Vecteur " + str(k))
		plt.show()
"""


fund = []
fund_damaged = []
for x in ev:
	if cs.is_fundamental(x):
		fund += [ x ]
for x in ev_damaged:
	if cs.is_fundamental(x):
		fund_damaged += [ x ]
		
if (len(fund) == len(fund_damaged)):
	for k in range(len(fund)):
		#plt.plot(pos,fund[k])
		#plt.plot(pos,fund_damaged[k])
		#plt.show()
		diff = minus_list(fund[k],fund_damaged[k])
		plt.plot(pos, diff)
		plt.show()
		dam, q = cs.max_derivate(diff)
		print("Damage on:" + str(dam) + " , quantification: " + str(q) )
		
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
"""

