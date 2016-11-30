#!/usr/bin/python

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import math

import parser
import sys
import json


###########################################################################################
###########################     Interpolation error    ####################################
###########################################################################################



#entree: liste de valeur de la FRF pour une frequence pour chaque capteur
#sortie: liste des erreurs au carre (une erreur par capteur)
def interpolation_error(complex_list):
	n = len(complex_list)
	x_list = [k for k in range(n)]
	error_list = []
	exc = 0
	for i in range(n):
		x = []
		y = []
		for j in range(n):
			if (j != i):
				x += [x_list[j]]
				y += [complex_list[j]]
		try:
			f = interp1d(x, y, kind = 'cubic')
			error_list += [ np.absolute(complex_list[i] - f(i)) ]
		except:
			#exc += 1
			error_list += [ 0 ]
	#print(exc)
	return error_list

#entree: liste de N listes de m elements ou
# N = nb de frequences
# m = nb de capteurs
#sortie: liste contenant l'erreur globale pour chaque capteur
def global_error(frf_list):
	nb_sensor = len(frf_list[0])
	error = [0 for i in range(nb_sensor) ]
	#k = 0
	for frf in frf_list:
		#k += 1
		#if (k>350):
		interpol = interpolation_error(frf)
		for i in range(nb_sensor):
			error[i] += interpol[i]**2
	for i in range(nb_sensor):
		error[i] = math.sqrt(error[i])
	return error


###########################################################################################
###########################     Reference calcul    #######################################
###########################################################################################



#entree: liste de liste de valeur de la FRF pour chaque capteur
#sortie: FRF moyenne
def create_ref(frf_list_ref):
	nb_frf = len(frf_list_ref)
	nb_sensor = len(frf_list_ref[0])
	nb_freq = len(frf_list_ref[0][0])

	frf_average = [ [0 for i in range(nb_sensor) ] for k in range(nb_freq) ]
	
	for i in range(nb_frf):
		for j in range(nb_sensor):
			for k in range(nb_freq):
				frf_average[k][j] = frf_average[k][j] + frf_list_ref[i][j][k]
	
	for i in range(nb_freq):
		for j in range(nb_sensor):
			frf_average[i][j] =  frf_average[i][j] / nb_frf
	
	std_dev = [ [0 for i in range(nb_sensor) ] for k in range(nb_freq) ]
	for i in range(nb_frf):
		for j in range(nb_sensor):
			for k in range(nb_freq):
				std_dev[k][j] += (np.absolute(frf_list_ref[i][j][k] - frf_average[k][j]))**2
				
	for i in range(nb_freq):
		for j in range(nb_sensor):
			std_dev[i][j] =  math.sqrt( std_dev[i][j] / (nb_frf - 1) )
	
	frf_inf = [ [0 for i in range(nb_sensor) ] for k in range(nb_freq) ]
	frf_sup = [ [0 for i in range(nb_sensor) ] for k in range(nb_freq) ]
	
	for j in range(nb_sensor):
			for k in range(nb_freq):
				frf_inf[k][j] = np.absolute(frf_average[k][j]) - std_dev[k][j]
				frf_sup[k][j] = np.absolute(frf_average[k][j]) + std_dev[k][j]
	
	return frf_average, frf_inf, frf_sup
	





###########################################################################################
###########################     Interpolation tests    ####################################
###########################################################################################


"""
#d = parser.get(sys.argv[1])
x =[1, 2, 3, 4, 5] #donnees
y = [1.+9.j, 1.+2.j, 1.+3.j, 1.+25.j, 2] #donnees

x2 =[1, 2, 3, 5] #donnees
y2 = [1.+9.j, 1.+2.j, 1.+3.j, 2] #donnees
#f = interp1d(x, y) #linear
#f2 = interp1d(x,y, kind = 'quadratic')
#f3 = interp1d(x2,y2, kind = 'cubic')
#t = f3(x)
for j in x:
	if(j>1 and j<len(x)) :  
		print "Suppression du capteur : " + str(j)
		x2 = list(x)
		y2 = list(y)
		del x2[j-1]
		del y2[j-1]
		f3 = interp1d(x2, y2, kind = 'cubic')
		for i in x :
			print "Erreur d'interpolation du capteur " + str(i) + " : " + str(np.absolute(y[i-1] - f3(i)))
			print "\n"
print "Pour x = 4, lineaire y = " + str(f(4)) + ", quadratique y = " + str(f2(4)) + " \net cubique = " + str(f3(4))
r, i = [], []
r2, i2 = [], []
for k in t:
	r += [ k.real ]
	i += [ k.imag ]

for k in y:
	r2 += [ k.real ]
	i2 += [ k.imag ]
plt.plot(r)
plt.plot(i)
plt.plot(r2)
plt.plot(i2)
plt.show()
"""



