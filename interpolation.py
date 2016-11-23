#!/usr/bin/python

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

import parser
import sys

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
