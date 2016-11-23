import frf
import matplotlib.pyplot as plt
import numpy
import tool
import parser
import simu_one_sensor as simu


CAPTVALUE = 4

def fibo(n):
    if n <= 1: return 1
    return(fibo(n-1) + fibo(n-2))

def frf_multiple_capter_file(inFile=None):
    pos = [ k for k in range(CAPTVALUE) ]
    mass = [ 100 for k in range(CAPTVALUE) ]
    stiffness = [ 100 for k in range(CAPTVALUE) ]
    N = CAPTVALUE
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


def frf_multiple_capter_simu(inFile=None):
    pos = [ k for k in range(CAPTVALUE) ]
    mass = [ 100 for k in range(CAPTVALUE) ]
    stiffness = [ 100 for k in range(CAPTVALUE) ]
    N = CAPTVALUE
    forces = simu.white_noise(100000)
    #simu: a activer seulement si on veut de nouvelles donnees
    measures = simu.simulation(pos,mass,stiffness,forces,1)
    #donnes de la simu
    forces = parser.get("F.txt")[:]
    measures = parser.get("Y.txt")[:]
    #
    res = []
    for i in range(CAPTVALUE):
        res.append(frf.frf(forces,tool.get_lines(measures,i)))
    if inFile == None:
        return res
    else:
        parser.writeValues(inFile,res)
        print("Values written in " + inFile)


def printPerCapter(res,time):
    n = len(res)
    delta = time/n
    plt.ion()
    for i in range(n):
        plt.plot(numpy.absolute(res[i]))
        plt.pause(delta)
        plt.clf()

def printPerFreq(res,time):
    n = len(res[0])
    delta = time/n/10
    plt.ion()
    for i in range(n):
        plt.plot(numpy.absolute(tool.get_lines(res,i)))
        plt.pause(delta)
        plt.clf()

printPerCapter(frf_multiple_capter_simu(),10)
