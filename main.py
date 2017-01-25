###########################################################################################
###########################     Imports    ################################################
###########################################################################################


import frf
import matplotlib.pyplot as plt
import numpy
import tool
import parser
import interpolation as itp
import experience

###########################################################################################
###########################     Variables    ##############################################
###########################################################################################

UNDAMAGED = "exper/expundamaged1.txt"
DAMAGED = "exper/expdamaged1.txt"

###########################################################################################
###########################     FRF computation    ########################################
###########################################################################################



def printPerCapter(res,freq=[1]):
    n = len(res)
    x = [ ((float(k)/len(res[0]))*freq[0]) for k in range(len(res[0])) ]
    for i in range(n):
        plt.plot(x,numpy.absolute(res[i]))
        plt.xlabel("Frequencies (Hz)")
        plt.ylabel("FRF")
        plt.legend()
        plt.show()

def printPerCapter2(res1,res2,time):
    n = len(res1)
    delta = time/n
    plt.ion()
    for i in range(n):
        plt.plot(numpy.absolute(res1[i]))
        plt.plot(numpy.absolute(res2[i]))
        plt.pause(delta)
        plt.clf()

def printPerFreq(res):
    n = len(res[0])
    for i in range(n):
        plt.plot(numpy.absolute(tool.get_lines(res,i)))
        plt.show()
