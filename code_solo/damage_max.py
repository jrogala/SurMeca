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
DAMAGE = 100# % of damage

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


def two_max_ind(tab):
	if tab[0] > tab[1]:
		a = 0
		va = tab[0]
		b = 1
		vb = tab[1]
	else:
		a = 1
		va = tab[1]
		b = 0
		vb = tab[0]
	for i in range(2,len(tab)):
		if tab[i] > vb:
			if tab[i] > va:
				b = a
				a = i
				vb = va
				va = tab[i]
			else:
				b = i
				vb = tab[i]
	return a, b, va, vb



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
	error_ref = itp.global_error(frf_ref)
	
	detect_damage1 = [ 0 for k in range(CAPTVALUE) ]
	detect_damage2 = [ 0 for k in range(CAPTVALUE) ]
	detect_damage1_inv = [ 0 for k in range(CAPTVALUE) ]
	detect_damage2_inv = [ 0 for k in range(CAPTVALUE) ]
	
	threshold1 = parser.get3("data/threshold1.txt")
	threshold2 = parser.get3("data/threshold2.txt")
	
	threshold1_inv = parser.get3("data/threshold1_inv.txt")
	threshold2_inv = parser.get3("data/threshold2_inv.txt")
	
	quantification1 = [ 0 for k in range(CAPTVALUE) ]
	quantification2 = [ 0 for k in range(CAPTVALUE) ]
	quantification1_inv = [ 0 for k in range(CAPTVALUE) ]
	quantification2_inv = [ 0 for k in range(CAPTVALUE) ]
	
	for k in range(MAKEREF):
		print(k)
		frf_damaged, acc = frf_multiple_sensors(2, acc)
		error_damaged = itp.global_error(frf_damaged)
		error_damaged_inv = itp.global_error(minus_list(frf_ref,frf_damaged))
		diff = minus_list(error_damaged, error_ref)
		#threshold1
		ind_max1, ind_max2, max1, max2 = two_max_ind(minus_list(diff,threshold1))
		if ind_max1 > 0:
			detect_damage1[ind_max1] = detect_damage1[ind_max1] + 1
			quantification1[ind_max1] = quantification1[ind_max1] + max1
		if ind_max2 > 0:
			detect_damage1[ind_max2] = detect_damage1[ind_max2] + 1
			quantification1[ind_max2] = quantification1[ind_max2] + max2
		#threshold2
		ind_max1, ind_max2, max1, max2 = two_max_ind(minus_list(diff,threshold2))
		if ind_max1 > 0:
			detect_damage2[ind_max1] = detect_damage2[ind_max1] + 1
			quantification2[ind_max1] = quantification2[ind_max1] + max1
		if ind_max2 > 0:
			detect_damage2[ind_max2] = detect_damage2[ind_max2] + 1
			quantification2[ind_max2] = quantification2[ind_max2] + max2
		#threshold_inv1
		ind_max1, ind_max2, max1, max2 = two_max_ind(minus_list(error_damaged_inv,threshold1_inv))
		if ind_max1 > 0:
			detect_damage1_inv[ind_max1] = detect_damage1_inv[ind_max1] + 1
			quantification1_inv[ind_max1] = quantification1_inv[ind_max1] + max1
		if ind_max2 > 0:
			detect_damage1_inv[ind_max2] = detect_damage1_inv[ind_max2] + 1
			quantification1_inv[ind_max2] = quantification1_inv[ind_max2] + max2
		#threshold_inv2
		ind_max1, ind_max2, max1, max2 = two_max_ind(minus_list(error_damaged_inv,threshold2_inv))
		if ind_max1 > 0:
			detect_damage2_inv[ind_max1] = detect_damage2_inv[ind_max1] + 1
			quantification2_inv[ind_max1] = quantification2_inv[ind_max1] + max1
		if ind_max2 > 0:
			detect_damage2_inv[ind_max2] = detect_damage2_inv[ind_max2] + 1
			quantification2_inv[ind_max2] = quantification2_inv[ind_max2] + max2
	
	parser.writeValues3("data/detect_damage_1_"+str(DAMAGE)+"percent_max.txt",detect_damage1)
	parser.writeValues3("data/detect_damage_2_"+str(DAMAGE)+"percent_max.txt",detect_damage2)
	parser.writeValues3("data/detect_damage_1_inv_"+str(DAMAGE)+"percent_max.txt",detect_damage1_inv)
	parser.writeValues3("data/detect_damage_2_inv_"+str(DAMAGE)+"percent_max.txt",detect_damage2_inv)
	
	print("Detect_damage writed in data/detect_damage_i_"+str(DAMAGE)+"percent.txt")
	
	
		
	for k in range(1,CAPTVALUE-1):
		quantification1[k] = (quantification1[k]/MAKEREF ) / threshold1[k]
		quantification2[k] = (quantification2[k]/MAKEREF) / threshold2[k]
		quantification1_inv[k] = (quantification1_inv[k]/MAKEREF) / threshold1_inv[k]
		quantification2_inv[k] = (quantification2_inv[k]/MAKEREF) / threshold2_inv[k]
	
	
	NOISE = 0
	parser.writeValues3("data/quantification1_"+str(DAMAGE)+"damage_"+str(NOISE)+"noise.txt",quantification1)
	parser.writeValues3("data/quantification2_"+str(DAMAGE)+"damage_"+str(NOISE)+"noise.txt",quantification2)
	parser.writeValues3("data/quantification1_inv_"+str(DAMAGE)+"damage_"+str(NOISE)+"noise.txt",quantification1_inv)
	parser.writeValues3("data/quantification2_inv_"+str(DAMAGE)+"damage_"+str(NOISE)+"noise.txt",quantification2_inv)
	
	print("Quantification writed in data/quantification_i_"+str(DAMAGE)+"damage_"+str(NOISE)+"noise.txt")
	
	
	for k in range(CAPTVALUE):
		if detect_damage1[k] > 0:
			print("Threshold 1: damage detected on sensor " + str(k) + "with quantification : " + str(quantification1[k]) )
	
	for k in range(CAPTVALUE):
		if detect_damage2[k] > 0:
			print("Threshold 2: damage detected on sensor " + str(k) + "with quantification : " + str(quantification2[k]) )
	
	for k in range(CAPTVALUE):
		if detect_damage1_inv[k] > 0:
			print("Threshold 1 inv: damage detected on sensor " + str(k) + "with quantification : " + str(quantification1_inv[k]) )
	
	for k in range(CAPTVALUE):
		if detect_damage2_inv[k] > 0:
			print("Threshold 2 inv: damage detected on sensor " + str(k) + "with quantification : " + str(quantification2_inv[k]) )
	
	
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



