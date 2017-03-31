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

NOISE = 0.04
CAPTVALUE = 20
SAMPLVALUE = 10000
MAKEREF = 100
DAMAGED_SENSOR = [7]#must be a list
DAMAGE = 33# % of damage

pos = [ k for k in range(CAPTVALUE) ]
z = [ 0 for k in range(CAPTVALUE) ]
mass = [ 100 for k in range(CAPTVALUE) ]


stiffness_undamaged = [ 100 for k in range(CAPTVALUE) ]
stiffness_damaged = [ 100 for k in range(CAPTVALUE) ]

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

def av_std_for_noise(l):# for add noise on measurement, l = [ [ a1, a2, a3], [a4, a5, a6] ... ]
	n = len(l)
	k = len(l[0])
	av = 0
	std = 0
	for i in range(n):
		for j in range(k):
			av += l[i][j]
	av = av/float(n)
	for i in range(n):
		for j in range(k):
			std += (l[i][j] - av)**2
	std = math.sqrt(std/float(n))
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

A_damaged, B_damaged, C_damaged, D_damaged, freq_useless = make_matrices(mass, stiffness_damaged)

#acc = [ [0 for k in range(2*n) ] ]


def simulation(typ,acc,std_noise):
	if (typ == 0):#init
		f = white_noise(SAMPLVALUE,n)
		m_stats = []
		for i in range(SAMPLVALUE):
			acc += [ np.dot(A,acc[-1]) + np.dot(B,f[i]) ]
			m_stats += [ np.dot(C,acc[-2]) + np.dot(D,f[i]) ]
		av, std = av_std_for_noise(m_stats)
		return [std], [], acc
	
	if (typ == 1):#no damage
		measurement_ref = []
		m = []
		f = white_noise(SAMPLVALUE,n)
		for i in range(SAMPLVALUE):
			acc += [ np.dot(A,acc[-1]) + np.dot(B,f[i]) ]
			m += [ np.dot(C,acc[-2]) + np.dot(D,f[i]) ]
		measurement_ref = m
		for i in range(SAMPLVALUE):
			for j in range(CAPTVALUE):
				measurement_ref[i][j] = measurement_ref[i][j] + NOISE*norm.rvs(size=1, loc = 0, scale = std_noise)[0]
		return measurement_ref, f, acc
	
	if (typ == 2):#damage
		measurement_damaged = []
		f_damaged = white_noise(SAMPLVALUE,n)
		for i in range(SAMPLVALUE):
			acc += [ np.dot(A_damaged,acc[-1]) + np.dot(B_damaged,f_damaged[i]) ]
			measurement_damaged += [ np.dot(C_damaged,acc[-2]) + np.dot(D_damaged,f_damaged[i]) ]
		for i in range(SAMPLVALUE):
			for j in range(CAPTVALUE):
				measurement_damaged[i][j] = measurement_damaged[i][j] + NOISE*norm.rvs(size=1, loc = 0, scale = std_noise)[0]
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

def frf_multiple_sensors(typ,acc,std):
	measurement, f, acc = simulation(typ,acc,std)
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
	s, _, acc = simulation(0, [ [0 for k in range(2*n) ] ], 0)
	std_noise = s[0]

	frf_ref, acc = frf_multiple_sensors(1, acc,std_noise)
	error_ref = itp.global_error(frf_ref)
	
	detect_damage1 = [ 0 for k in range(CAPTVALUE) ]
	detect_damage2 = [ 0 for k in range(CAPTVALUE) ]
	detect_damage1_inv = [ 0 for k in range(CAPTVALUE) ]
	detect_damage2_inv = [ 0 for k in range(CAPTVALUE) ]
	
	threshold1 = parser.get3("data/threshold1_noise"+str(NOISE)+".txt")
	threshold2 = parser.get3("data/threshold2_noise"+str(NOISE)+".txt")
	threshold1_inv = parser.get3("data/threshold1_inv_noise"+str(NOISE)+".txt")
	threshold2_inv = parser.get3("data/threshold2_inv_noise"+str(NOISE)+".txt")
	
	for k in range(MAKEREF):
		print(k)
		frf_damaged, acc = frf_multiple_sensors(2, acc,std_noise)
		error_damaged = itp.global_error(frf_damaged)
		error_damaged_inv = itp.global_error(minus_list(frf_ref,frf_damaged))
		diff = minus_list(error_damaged, error_ref)
		for x in range(CAPTVALUE):
			if diff[x] > threshold1[x]:
				detect_damage1[x] = detect_damage1[x] + 1
			if diff[x] > threshold2[x]:
				detect_damage2[x] = detect_damage2[x] + 1
			if error_damaged_inv[x] > threshold1_inv[x]:
				detect_damage1_inv[x] = detect_damage1_inv[x] + 1
			if error_damaged_inv[x] > threshold2_inv[x]:
				detect_damage2_inv[x] = detect_damage2_inv[x] + 1
	
	parser.writeValues3("data/detect_damage_1_"+str(DAMAGE)+"percent_noise"+str(NOISE)+".txt",detect_damage1)
	parser.writeValues3("data/detect_damage_2_"+str(DAMAGE)+"percent_noise"+str(NOISE)+".txt",detect_damage2)
	parser.writeValues3("data/detect_damage_1_inv_"+str(DAMAGE)+"percent_noise"+str(NOISE)+".txt",detect_damage1_inv)
	parser.writeValues3("data/detect_damage_2_inv_"+str(DAMAGE)+"percent_noise"+str(NOISE)+".txt",detect_damage2_inv)
	print("Detect_damage writed in data/detect_damage_i_"+str(DAMAGE)+"percent_noise"+str(NOISE)+".txt")
	
	plt.plot(detect_damage1, label = 'Simple Threshold')
	plt.plot(detect_damage2, label = 'With Gaussiennes')
	plt.plot(detect_damage1_inv, label = 'Simple Threshold Inv')
	plt.plot(detect_damage2_inv, label = 'With Gaussiennes Inv')
	plt.title("Damage: "+str(DAMAGE)+"%")
	plt.legend()
	plt.show()

###########################################################################################
###########################     Script    #################################################
###########################################################################################

experience()



