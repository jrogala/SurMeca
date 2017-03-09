################################################
############# import ###########################
################################################

import numpy as np
import math
import cmath
import scipy.linalg
from scipy.stats import norm
import scipy.signal
import tool
import print_functions as pr
import interpolation as itp
import matplotlib.pyplot as plt
import parser
import random

################################################
############# variables ########################
################################################


CAPTVALUE = 20
SAMPLVALUE = 10000
MAKEREF = 100
DAMAGED_SENSOR = [7]#must be a list
EXCITED_SENSOR = 0
DAMAGE = 0# % of damage

pos = [ k for k in range(CAPTVALUE) ]
z = [ 0 for k in range(CAPTVALUE) ]
mass = [ 100 for k in range(CAPTVALUE) ]


indic = ""
stiffness_undamaged = [ 100 for k in range(CAPTVALUE) ]
stiffness_damaged = [ 100 for k in range(CAPTVALUE) ]


#stiffness_undamaged = [ 80+random.randint(1,40) for k in range(CAPTVALUE) ]
#stiffness_damaged = [ stiffness_undamaged[k] for k in range(CAPTVALUE) ]




#for noise
mean = 0
standard_deviation = 10

################################################
############# functions ########################
################################################


def av_std(l):# l = [ [ a1, a2, a3], [a4, a5, a6] ... ]
	n = len(l)
	k = len(l[0])
	av = [ 0 for i in range(k) ]
	std = [ 0 for i in range(k) ]
	for i in range(n):
		for j in range(k):
			av[j] = av[j] + l[i][j]
	for i in range(k):
		av[i] = av[i]/float(n)
	for i in range(n):
		for j in range(k):
			std[j] = std[j] + (l[i][j] - av[j])**2
	for i in range(k):
		std[i] = math.sqrt(std[i]/float(n))
	return av, std

def gauss(av,std,start,end,nb_pts):
	x = [ start + float((end-start)*k)/nb_pts for k in range(nb_pts)]
	g = [ (1/(std*math.sqrt(2*math.pi)))*math.exp(-(x[k] - av)**2/(2*std**2)) for k in range(nb_pts) ]
	return x, g

def matrix_M(mass):#make the mass matrix
	return np.diag(mass)

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

def matrix_C(M,K,a,b):#make the C matrix
	return a*M + b*K

def minus_list(l,m):
	n = len(l)
	r = [ (l[k] - m[k]) for k in range(n)]
	return r

def add_list(l,m):
	n = len(l)
	r = [ (l[k] + m[k]) for k in range(n)]
	return r

def add_list2(l,m):
	n = len(l)
	k = len(l[0])
	r = [ [ 0 for i in range(k) ] for j in range(n) ]
	for j in range(n):
		for i in range(k):
			r[j][i] = l[j][i] + m[j][i]
	return r
	
##### white noise ######


def white_noise(n,nb_sensor):
	noise = []
	for i in range(n):
		noise += [ norm.rvs(size=nb_sensor, loc = mean, scale = standard_deviation) ]
	return noise


def make_matrices(mass, stiffness):
#Make matrices and sample frequency
	n = CAPTVALUE
	M = matrix_M(mass)
	K = matrix_K(stiffness)
	C = matrix_C(M,K,0.005,0.005)
	
	Minv = np.linalg.inv(M)
	MinvK = np.dot(Minv,K)
	MinvC = np.dot(Minv,C)
	
	I = np.eye(2*n)
	A = np.zeros((2*n,2*n))
	B = np.zeros((2*n,n))
	C = np.zeros((n,2*n))
	D = np.linalg.inv(M)
	for i in range(n):
		for j in range(n):
			A[n+i][j] = - MinvK[i][j]
			A[n+i][n+j] = - MinvC[i][j]
			B[n+i][j] = Minv[i][j]
			C[i][j] = - MinvK[i][j]
			C[i][n+j] = - MinvC[i][j]
		A[i][n+i] = 1
#now, we calculate the time of measure delta_t
	eigenvalues, eigenvectors = np.linalg.eig(A)
	frq = []
	for i in eigenvalues:
		frq += [abs(i)/(2*math.pi)]
	max_frq = max(frq)
	#print( frq )
	f_sampl = (2.5)*max_frq  #Shannon
	delta_t = 1/f_sampl
	Ad = scipy.linalg.expm(delta_t*A)
	Bd = np.dot((Ad - I), np.dot( np.linalg.inv(A) , B ) )
	return Ad, Bd, C, D, f_sampl



################################################
############# Simulation #######################
################################################


n = CAPTVALUE
A, B, C, D, sampling_freq = make_matrices(mass, stiffness_undamaged)

for x in DAMAGED_SENSOR:
	stiffness_damaged[x] = int(float(stiffness_undamaged[x])*float(100-DAMAGE)/100)

A_damaged, B_damaged, C_useless, D_useless, freq_useless = make_matrices(mass, stiffness_damaged)

#acc = [ [0 for k in range(2*n) ] ]


def simulation(typ,acc):
	if (typ == 0):#init
		f = white_noise(SAMPLVALUE,n)
		for i in range(SAMPLVALUE):
			acc += [ np.dot(A,acc[-1]) + np.dot(B,f[i]) ]
		return [], [], acc
	
	if (typ == 1):#no damage
		measurement_ref = []
		m = []
		f = white_noise(SAMPLVALUE,n)
		for i in range(SAMPLVALUE):
			acc += [ np.dot(A,acc[-1]) + np.dot(B,f[i]) ]
			m += [ np.dot(C,acc[-2]) + np.dot(D,f[i]) ]
		measurement_ref = m
		return measurement_ref, f, acc
	
	if (typ == 2):#damage
		measurement_damaged = []
		f_damaged = white_noise(SAMPLVALUE,n)
		for i in range(SAMPLVALUE):
			acc += [ np.dot(A_damaged,acc[-1]) + np.dot(B_damaged,f_damaged[i]) ]
			measurement_damaged += [ np.dot(C,acc[-2]) + np.dot(D,f_damaged[i]) ]
		return measurement_damaged, f_damaged, acc

	
	
################################################
############# FRF ##############################
################################################

NPERSEGVALUE = 1024

def frf(i,o,f,est = "H1"):
    """
    Return the frf of input/output
    """
    #ifft = np.fft.fft(i)
    #offt = np.fft.fft(o)
    fo, oicsd = scipy.signal.csd(o,i,fs=f,nperseg = NPERSEGVALUE)
    fi, iwelch = scipy.signal.welch(i,fs=f,nperseg = NPERSEGVALUE)
    return (oicsd/iwelch), fo
	
################################################
############# FRF calcul #######################
################################################

def frf_multiple_sensors(typ,acc):
	measurement, f, acc = simulation(typ,acc)
	if (typ != 0):
		res = []
		res_freq = []
		for j in range(CAPTVALUE):		
			tmp = []
			freq = []
			for i in range(CAPTVALUE):
				v_frf, f_frf = frf(tool.get_lines(f,j),tool.get_lines(measurement,i),sampling_freq)
				tmp.append(v_frf)
				freq.append(f_frf)
			if (j == 0):
				test_v = list(tmp)
			else:
				test_v = add_list2(test_v, tmp)
		res = test_v
		res_freq = freq
		return np.transpose(res), acc




###########################################################################################
###########################     Experience    #############################################
###########################################################################################

def experience():
	_, _, acc = simulation(0, [ [0 for k in range(2*n) ] ])
	

	frf_ref, acc = frf_multiple_sensors(1, acc)
	#error_ref = itp.global_error(frf_ref)
	error_list = []
	
	for k in range(MAKEREF):
		print(k)
		frf_undamaged, acc = frf_multiple_sensors(1, acc)
		error_undamaged = itp.global_error(minus_list(frf_ref,frf_undamaged))
		error_list += [error_undamaged]

	threshold = [ 0 for k in range(CAPTVALUE)]
	
	for k in range(1,CAPTVALUE-1):
		e_list = tool.get_lines(error_list,k)
		threshold[k] = max(e_list)
		if (0):
			plt.hist(e_list, bins = 10)
			plt.title("Error repartition for sensor: " + str(k))
			plt.legend()
			plt.show()
	
	parser.writeValues3("data/threshold1_inv.txt",threshold)
	print("Threshold writed in data/threshold1_inv.txt")

###########################################################################################
###########################     Script    #################################################
###########################################################################################

experience()



