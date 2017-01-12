###########################################################################################
###########################     Imports    ################################################
###########################################################################################


import frf
import matplotlib.pyplot as plt
import numpy
import tool
import parser
#import simu_one_sensor as simu
import simu

###########################################################################################
###########################     Variables    ##############################################
###########################################################################################


CAPTVALUE = simu.CAPTVALUE
SAMPLVALUE = simu.SAMPLVALUE
MAKEREF = simu.MAKEREF
DAMAGED_SENSOR = simu.DAMAGED_SENSOR
DAMAGE = simu.DAMAGE


###########################################################################################
###########################     FRF calculs    ############################################
###########################################################################################


def fibo(n):
    if n <= 1: return 1
    return(fibo(n-1) + fibo(n-2))

def frf_multiple_capter_file(inFile=None):
    #donnes du prof
    forces = parser.get("donnees/F.txt")[:]
    measures = parser.get("donnees/Y.txt")[:]
    #
    res = []
    for i in range(CAPTVALUE):
        res.append(frf.frf(forces,tool.get_lines(measures,i)))
    if inFile == None:
        return res
    else:
        parser.writeValues(inFile,res)
        print("Values written in " + inFile)
        


def frf_multiple_capter_simu_undamaged(inFile=None, fichier_sampl = "f_sampl_undamaged.txt"):
    print("*************************\n Simulation for undamaged state \n *************************")
    #simu: a activer seulement si on veut de nouvelles donnees
    forces = simu.white_noise(SAMPLVALUE,CAPTVALUE)
    acc, measures, f_sampl = simu.simulation(forces,"F_undamaged.txt","Y_undamaged.txt", fichier_sampl)
    #donnes de la simu
    forces = parser.get("F_undamaged.txt")[:]
    measures = parser.get("Y_undamaged.txt")[:]
    #
    res = []
    for i in range(CAPTVALUE):
        res.append(frf.frf(tool.get_lines(forces,i),tool.get_lines(measures,i)))
    if (inFile == None):
        return res
    else:
        parser.writeValues(inFile,res)
        print("Values written in " + inFile)
        #return res


def frf_multiple_capter_simu_damaged(inFile=None, fichier_sampl = "f_sampl_damaged.txt"):
    print("*************************\n Simulation for damaged state \n *************************")
    #simu: a activer seulement si on veut de nouvelles donnees
    forces = simu.white_noise(SAMPLVALUE,CAPTVALUE)
    acc, measures, f_sampl = simu.simulation(forces,"F_damaged.txt","Y_damaged.txt", fichier_sampl, damaged = "f_sampl_undamaged.txt")
    #donnes de la simu
    forces = parser.get("F_damaged.txt")[:]
    measures = parser.get("Y_damaged.txt")[:]
    #
    
    res = []
    for i in range(CAPTVALUE):
        res.append(frf.frf(tool.get_lines(forces,i),tool.get_lines(measures,i)))
    if inFile == None:
        return res
    else:
        parser.writeValues(inFile,res)
        print("Values written in " + inFile)
        #return res
        
        
def frf_multiple_capter_simu_ref(inFile=None, fichier_sampl = "f_sampl_ref.txt"):
    print("*************************\n Simulation for reference state \n *************************")
    for i in range(MAKEREF):
    	frf_multiple_capter_simu_undamaged("FRF_ref" + str(i) + ".txt")




###########################################################################################
###########################     Print functions    ########################################
###########################################################################################




def printPerCapter(res,time,freq=[1]):
    n = len(res)
    x = [ ((float(k)/len(res[0]))*freq[0]) for k in range(len(res[0])) ]
    print(x)
    print(freq)
    delta = time/n
    #plt.ion()
    for i in range(n):
        plt.plot(x,numpy.absolute(res[i]))
        plt.xlabel("Frequencies (Hz)")
        plt.ylabel("FRF")
        #plt.pause(delta)
        #plt.clf()
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

def printPerFreq(res,time):
    n = len(res[0])
    delta = time/n/10
    #plt.ion()
    for i in range(n):
        plt.plot(numpy.absolute(tool.get_lines(res,i)))
        #plt.pause(delta)
        #plt.clf()
        plt.show()
        
        
#frf_multiple_capter_simu_ref("FRF_ref_")        
#frf_multiple_capter_simu("FRF_damage.txt")
#printPerCapter(frf_multiple_capter_simu_undamaged(),10, freq = parser.get("f_sampl_undamaged.txt"))
#printPerCapter2(frf_multiple_capter_simu_damaged(),frf_multiple_capter_simu_undamaged(),10)
