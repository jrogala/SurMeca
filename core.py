import numpy as np
import math
import cmath
import scipy.linalg
from scipy.stats import norm
import pickle

class Simulation(object):
    """docstring for Experience.
    Une experience entiere."""
    def __init__(self, name=""):
        if name == "":
            self.masse = []
            self.capteur = []
            self.accel = []
            self.force = []
            self.echantillon = 0
            self.noise = []
        else:
            self.load(name)
    def load(self,name):
        with open("exper/" + name,'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
    def save(self,name):
        with open("exper/" + name,'wb') as f:
            pickle.dump(self.__dict__,f,2)


    def experiment(self,mass,stiffness,echantillon):

        #for noise
        mean = 0
        standard_deviation = 10

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
                K[i+1][i] = - stiffness[i+1]
                K[i][i+1] = - stiffness[i+1]
            return K

        def matrix_C(M,K,a,b):#make the C matrix
            return a*M + b*K


        ##### white noise ######


        def white_noise(n,nb_sensor):
            noise = []
            for i in range(n):
                noise += [ norm.rvs(size=nb_sensor, loc = mean, scale = standard_deviation) ]
            return noise


        def make_matrices(mass, stiffness):
        #Make matrices and sample frequency
            n = len(mass)
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

        def simulation():
            n = len(mass)
            A, B, C, D, freq = make_matrices(mass, stiffness)

            A_damaged, B_damaged, _, _, _ = make_matrices(mass, stiffness)

            acc = [ [0 for k in range(2*n) ] ]
            measurement = []
            m = []
            f = white_noise(echantillon,n)
            for i in range(echantillon):
                acc += [ np.dot(A,acc[-1]) + np.dot(B,f[i]) ]

            f = white_noise(echantillon,n)
            for i in range(echantillon):
                acc += [ np.dot(A,acc[-1]) + np.dot(B,f[i]) ]
                m += [ np.dot(C,acc[-2]) + np.dot(D,f[i]) ]
            return m,f
        self.accel,self.force = simulation()
