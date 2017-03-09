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



NB_EXP = 20
CAPTVALUE = 20
SAMPLVALUE = 10000
MAKEREF = 10
DAMAGED_SENSOR = [13]#must be a list
EXCITED_SENSOR = 0
DAMAGE = 5# % of damage

pos = [ k for k in range(CAPTVALUE) ]
z = [ 0 for k in range(CAPTVALUE) ]
mass = [ 100 for k in range(CAPTVALUE) ]


indic = ""
stiffness_undamaged = [ 100 for k in range(CAPTVALUE) ]
stiffness_damaged = [ 100 for k in range(CAPTVALUE) ]


#stiffness_undamaged = [ 50+10*random.randint(1,10) for k in range(CAPTVALUE) ]
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


##### white noise ######

"""
def white_noise(n,nb_sensor):
	noise = []
	for i in range(n):
		noise += [ norm.rvs(size=nb_sensor, loc = mean, scale = standard_deviation) ]
	return noise
"""

def white_noise(n,nb_sensor):
	noise = []
	for i in range(n):
		noise += [ norm.rvs(size=nb_sensor, loc = mean, scale = standard_deviation) ]
	for i in range(n):
		for k in range(nb_sensor):
			if (k != EXCITED_SENSOR):
				noise[i][k] = 0
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

def simulation(damage,d):
	n = CAPTVALUE
	A, B, C, D, freq = make_matrices(mass, stiffness_undamaged)
	
	for x in damage:
		stiffness_damaged[x] = int(float(stiffness_undamaged[x])*float(100-d)/100)
	
	#print(stiffness_damaged)
	
	A_damaged, B_damaged, C_useless, D_useless, freq_useless = make_matrices(mass, stiffness_damaged)

	acc = [ [0 for k in range(2*n) ] ]
	measurement_ref = []
	measurement_damaged = []
	
	f = white_noise(SAMPLVALUE,n)
	for i in range(SAMPLVALUE):
		acc += [ np.dot(A,acc[-1]) + np.dot(B,f[i]) ]
	
	m = []
	f = white_noise(SAMPLVALUE,n)
	for i in range(SAMPLVALUE):
		acc += [ np.dot(A,acc[-1]) + np.dot(B,f[i]) ]
		m += [ np.dot(C,acc[-2]) + np.dot(D,f[i]) ]
	measurement_ref = m
	
	#now, there is a damage
	f_damaged = white_noise(SAMPLVALUE,n)
	for i in range(SAMPLVALUE):
		acc += [ np.dot(A_damaged,acc[-1]) + np.dot(B_damaged,f_damaged[i]) ]
		measurement_damaged += [ np.dot(C,acc[-2]) + np.dot(D,f_damaged[i]) ]
	return measurement_ref, measurement_damaged, f, f_damaged, freq
	
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
############# FRF theorique ####################
################################################

def matrix_H(M, K, C, f):#f: frequency
	omega = 2*f*math.pi
	s = 1j*omega
	H = np.linalg.inv((s**2)*M+s*C+K)
	return H

def frf_theo(freq, col, M, K, C):
	H = matrix_H(M, K, C, freq)
	res = []
	for k in range(CAPTVALUE):
		res += [ H[k][col] ]
	return res
	
################################################
############# FRF calcul #######################
################################################

def frf_multiple_sensors(inFileRef, inFileInf, inFileSup, inFileDamage,damage=DAMAGED_SENSOR,d=0):
	#print("*************************\n Simulation \n*************************")
	measurement, measurement_damaged, f, f_damaged, sampling_freq = simulation(DAMAGED_SENSOR,d)
	#print("*************************\n reference FRF computation \n*************************")
	res = []
	res_freq = []
	#for k in range(MAKEREF):
	tmp = []
	freq = []
	for i in range(CAPTVALUE):
		v_frf, f_frf = frf(tool.get_lines(f,EXCITED_SENSOR),tool.get_lines(measurement,i),sampling_freq)#[k] with MAKEREF
		tmp.append(v_frf)
		freq.append(f_frf)
	res = tmp
	res_freq = freq
	#res += [tmp]
	#res_freq += [freq]
	#frf_average, frf_inf, frf_sup = itp.create_ref(res)
	parser.writeValues(inFileRef,np.transpose(res))
	#print("Values written in " + inFileRef)
	#parser.writeValues(inFileInf,frf_inf)
	#print("Values written in " + inFileInf)
	#parser.writeValues(inFileSup,frf_sup)
	#print("Values written in " + inFileSup)
        
	#print("*************************\n damaged FRF computation \n *************************")
	res = []
	for i in range(CAPTVALUE):
		v_frf, f_frf = frf(tool.get_lines(f_damaged,EXCITED_SENSOR),tool.get_lines(measurement_damaged,i),sampling_freq)#[k] with MAKEREF
		res.append(v_frf)
	parser.writeValues(inFileDamage,np.transpose(res))
	#print("Values written in " + inFileDamage)
	return np.transpose(res_freq)




###########################################################################################
###########################     Experience    #############################################
###########################################################################################

def experience(dam=0):
	freqs = frf_multiple_sensors("data/FRF_ref"+indic+".txt","data/FRF_inf"+indic+".txt","data/FRF_sup"+indic+".txt","data/FRF_damaged"+indic+".txt",d=dam)
	#print(freqs)
	ref = parser.get2("data/FRF_ref"+indic+".txt")
	#inf = parser.get2("data/FRF_inf"+indic+".txt")
	#sup = parser.get2("data/FRF_sup"+indic+".txt")
	#frf_damaged = numpy.transpose(parser.get2("data/FRF_damaged"+indic+".txt"))
	frf_damaged = parser.get2("data/FRF_damaged"+indic+".txt")
	
	#M = matrix_M(mass)
	#K = matrix_K(stiffness_undamaged)
	#C = matrix_C(M,K,0.005,0.005)
	n = len(ref)
	if(0):
		for i in range(n):
			if (i%10 == 0):
				#print(freqs[i])
				freq = freqs[i][0]
				theo = frf_theo(freq, EXCITED_SENSOR, M, K, C)
				plt.plot(np.absolute(theo), label = "Theorical")
				plt.plot(np.absolute(ref[i]), label = "Undamaged")
				plt.plot(np.absolute(frf_damaged[i]), label = "Damaged")
				plt.xlabel('Sensors')
				plt.ylabel('FRF')
				plt.title('freq: ' + str(freq))
				plt.legend()
				#plt.pause(delta)
				#plt.clf()
				plt.show()
	
	
	error_ref = itp.global_error(ref)
	error_damaged = itp.global_error(frf_damaged)
	diff = minus_list(error_damaged, error_ref)
	if(0):
		plt.plot(pos,diff)
		plt.plot(pos,z)
		plt.show()

	#pr.printRef(ref,inf,sup)
	#pr.printAll(ref,inf,sup, frf_damaged)
	#e = itp.if_out(inf,sup,frf_damaged)
	#pr.print_pos(e)
	#pr.print_minus(ref,frf_damaged)
	return diff


###########################################################################################
###########################     Script    #################################################
###########################################################################################


avg_undamaged, avg_damaged = [ 0 for k in range(CAPTVALUE) ], [ 0 for k in range(CAPTVALUE) ]
std_undamaged, std_damaged = [ 0 for k in range(CAPTVALUE) ], [ 0 for k in range(CAPTVALUE) ]

l_damaged, l_undamaged = [], []

for k in range(NB_EXP):
	print("exp: " + str(k))
	l1 = experience(dam=0)
	l2 = experience(dam=DAMAGE)
	l_undamaged += [l1]
	l_damaged += [l2]

av_undamaged, std_undamaged = av_std(l_undamaged)
av_damaged, std_damaged = av_std(l_damaged)

"""
print(av_undamaged)
print(av_damaged)
print(std_undamaged)
print(std_damaged)
"""

for k in range(1,CAPTVALUE-1):
	m = min( av_undamaged[k] - 3* std_undamaged[k], av_damaged[k] - 3* std_damaged[k])
	M = max( av_undamaged[k] + 3* std_undamaged[k], av_damaged[k] + 3* std_damaged[k])
	nb_p = 1000
	x1, g_undamaged = gauss(av_undamaged[k], std_undamaged[k], m, M, nb_p)
	x2, g_damaged = gauss(av_damaged[k], std_damaged[k], m, M, nb_p)
	plt.plot(x1, g_undamaged, label = "Undamaged")
	plt.plot(x2, g_damaged, label = "Damaged")
	plt.xlabel('Interpolation error')
	plt.title('Sensor ' + str(k))
	plt.legend()
	plt.show()



