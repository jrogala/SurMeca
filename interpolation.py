#!/usr/bin/python

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

import parser
import sys
import json

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

"""
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

#entree: liste de valeur de la FRF pour une frequence pour chaque capteur
#sortie: liste des erreurs au carre (une erreur par capteur)
def interpolation_error(complex_list):
	n = len(complex_list)
	x_list = [k for k in range(n)]
	error_list = []
	for i in range(n):
		x = []
		y = []
		for j in range(n):
			if (j != i):
				x += [x_list[j]]
				y += [complex_list[j]]
				f = interp1d(x, y, kind = 'cubic')
				error_list += [ (np.absolute(complex_list[i] - f(i)))**2 ]
	return error_list

#entree: liste de N listes de m elements ou
# N = nb de frequences
# m = nb de capteurs
#sortie: liste contenant l'erreur globale pour chaque capteur
def global_error(frf_list):
	nb_sensor = len(frf_list[0])
	error = []
	for i in range(nb_sensor):
		e = 0
		for frf in frf_list:
			interpol = interpolation_error(frf)
			e += interpol[i]
		error += [math.sqrt(e)]
	return error



def create_ref(frf_ref,nb_frf):
	frf_list = []
	for i in range(nb_frf):
		frf_list += [parser.get2(frf_ref + str(i) + ".txt")]
	


create_ref("FRF_ref_",10)












