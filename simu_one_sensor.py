################################################
############# import ###########################
################################################



import numpy as np
import math
import cmath
import scipy.linalg
from scipy.stats import norm


################################################
############# functions ########################
################################################



def matrix_M(mass):#make the mass matrix
	return np.diag(mass)

def matrix_K(stiffness):#make the stiffness matrix
	n = len(stiffness)
	K = np.diag(stiffness)
	for i in range(n-1):
		K[i][i] += stiffness[i+1]
		K[i+1][i] = stiffness[i+1]
		K[i][i+1] = stiffness[i+1]
	return K

def matrix_C(M,K,a,b):#make the C matrix
	return a*M + b*K


def write_F(forces):
	f = open("F.txt",'w')
	for m in forces:
		p = 0#power
		if (m > 0):
			f.write(" ")
			nb = 9
		else:
			nb = 10
		if abs(m) > 0.1:
			while( abs(m) > 10 ):
				m /= 10
				p += 1
			f.write( "  " + str(m)[:nb] + "e+0" + str(p) + '\n')
		elif m == 0:
			f.write( "  " + "0.0000000" + "e+0" + str(p) + '\n')
		else:
			while( abs(m) < 0.1 ):
				m *= 10
				p += 1
			f.write( "  " + str(m)[:nb] + "e-0" + str(p) + '\n')
	f.close()




def write_Y(measures):
	f = open("Y.txt",'w')
	for acc in measures:
		for m in acc:
			p = 0#power
			if (m > 0):
				f.write(" ")
				nb = 9
			else:
				nb = 10
			if abs(m) > 0.1:
				while( abs(m) > 10 ):
					m /= 10
					p += 1
				f.write( "  " + str(m)[:nb] + "e+0" + str(p) + '\t')
			elif m == 0:
				f.write( "   " + "0.0000000" + "e+0" + str(p) + '\t')
			else:
				while( abs(m) < 0.1 ):
					m *= 10
					p += 1
				f.write( "  " + str(m)[:nb] + "e-0" + str(p) + '\t')
		f.write('\n')
	f.close()





def simulation(pos,mass,stiffness,f, col):
#pos: list of the positions of masses
#mass: list of mass
#stiffness: list of stiffness
#f: forces
#col: sensor

#return two lists:
#	-the first with velocity and acceleration of each mass
#   -the second with acceleration measured


#first, we make all matrixes

	N = len(f)
	n = len(mass)
	M = matrix_M(mass)
	K = matrix_K(stiffness)
	C = matrix_C(M,K,0.005,0.005)
	#print( M )
	#print( K )
	#print( C )
	
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
	"""
	print A
	print B
	print C
	print D
	print MinvK
	"""
	
	
#now, we calculate the time of measure delta_t
	
	eigenvalues, eigenvectors = np.linalg.eig(A)
	"""
	eigenvaluesMK, eigenvectorsMK = np.linalg.eig(MinvK)
	
	frqMK = []
	for i in eigenvaluesMK:
		frqMK += [ math.sqrt(abs(i))/(2*math.pi) ]
	"""
	frq = []
	for i in eigenvalues:
		frq += [abs(i)/(2*math.pi)]
	
	print( "Frequences du systeme:" )
	print( frq )
	#print( "Frequences MK:" )
	#print( frqMK )
	
	max_frq = max(frq)
	
	omega_max = max_frq*(2*math.pi)
	
	f_sampl = (2.5)*max_frq  #Shannon
	
	lambda_max = eigenvalues[ frq.index( max(frq) ) ]
	
	absorption = - (lambda_max.real) / omega_max
	
	absorption =  - eigenvalues.real / np.abs(eigenvalues)
	
	print "Coefficient d'amortissement: "
	print absorption
	
	delta_t = 1/f_sampl
	
	print "Frequence d'echantillonage (Hz): "
	print( f_sampl )
	
#simulation
	
	Ad = scipy.linalg.expm(delta_t*A)
	Bd = np.dot((Ad - I), np.dot( np.linalg.inv(A) , B ) )
	
	Bd_col = np.zeros(2*n)
	D_col = np.zeros(n)
	for i in range(2*n):
		Bd_col[i] = Bd[i][col]
	for i in range(n):
		D_col[i] = D[i][col]
	"""
	print(Bd)
	print(Bd_col)
	print(D)
	print(D_col)
	"""
	
	acc = [ [0 for k in range(2*n) ] ]
	measures = []
	
	for i in range(N):
		acc += [ np.dot(Ad,acc[-1]) + f[i]*Bd_col ]
		measures += [ np.dot(C,acc[-2]) + f[i]*D_col ]
	
	"""
	eigenvalues, eigenvectors = np.linalg.eig(A)
	
	print np.abs(eigenvalues / (2*math.pi))
	
	eigenvalues, eigenvectors = np.linalg.eig(Ad)
	
	eig_log = []
	for e in eigenvalues:
		eig_log += [ cmath.log(e) ]
	
	print( np.abs( eig_log/ delta_t ) / (2*math.pi) )
	"""
	print(measures[N-1])
	
	write_F(f)
	print("F.txt created")
	write_Y(measures)
	print("Y.txt created")
	
	return acc, measures








##### white noise ######

mean = 0
standard_deviation = 1

def white_noise(n):
	noise = []
	for i in range(n):
		noise += [ norm.rvs(loc = mean, scale = standard_deviation) ]
	return noise



#print( norm.rvs(loc = mean, scale = standard_deviation, size = 5) )



################################################
############# simulation #######################
################################################



N = 4

pos = [ k for k in range(N) ]
mass = [ 1 for k in range(N) ]
stiffness = [ 100 for k in range(N) ]

forces = white_noise(200)

acc, measures = simulation(pos,mass,stiffness,forces,1)


##### write #######

			
"""
f = open("velocity",'w')
g = open("acceleration",'w')
for i in acc:
	for k in range(N):
		f.write(str(i[k]) + " ")
		g.write(str(i[k+N]) + " ")
	f.write("\n")
	g.write("\n")
f.close()
g.close()

f = open("acceleration_measured",'w')
for i in measures:
	for k in range(N):
		f.write(str(i[k]) + " ")
	f.write("\n")
f.close()


f = open("noise",'w')
for i in forces:
	for k in range(N):
		f.write(str(i[k]) + " ")
	f.write("\n")
f.close()

"""


