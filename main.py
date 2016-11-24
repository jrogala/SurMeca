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


def frf_multiple_capter_simu_undamaged(inFile=None):
    print("*************************\n Simulation for undamaged state \n *************************")
    pos = [ k for k in range(CAPTVALUE) ]
    mass = [ 100 for k in range(CAPTVALUE) ]
    stiffness = [ 100 for k in range(CAPTVALUE) ]
    N = CAPTVALUE
    #simu: a activer seulement si on veut de nouvelles donnees
    forces = simu.white_noise(10000)
    measures = simu.simulation(pos,mass,stiffness,forces,1,"F_undamaged.txt","Y_undamaged.txt")
    #donnes de la simu
    forces = parser.get("F_undamaged.txt")[:]
    measures = parser.get("Y_undamaged.txt")[:]
    #
    
    res = []
    for i in range(CAPTVALUE):
        res.append(frf.frf(forces,tool.get_lines(measures,i)))
    if inFile == None:
        return res
    else:
        parser.writeValues(inFile,res)
        print("Values written in " + inFile)


def frf_multiple_capter_simu_damaged(inFile=None):
    print("*************************\n Simulation for damaged state \n *************************")
    pos = [ k for k in range(CAPTVALUE) ]
    mass = [ 100 for k in range(CAPTVALUE) ]
    stiffness = [ 100 for k in range(CAPTVALUE) ]
    stiffness[1] = 50
    N = CAPTVALUE
    #simu: a activer seulement si on veut de nouvelles donnees
    forces = simu.white_noise(10000)
    measures = simu.simulation(pos,mass,stiffness,forces,1,"F_damaged.txt","Y_damaged.txt")
    #donnes de la simu
    forces = parser.get("F_damaged.txt")[:]
    measures = parser.get("Y_damaged.txt")[:]
    #
    
    res = []
    for i in range(CAPTVALUE):
        res.append(frf.frf(forces,tool.get_lines(measures,i)))
    if inFile == None:
        return res
    else:
        parser.writeValues(inFile,res)
        print("Values written in " + inFile)
        
        
def frf_multiple_capter_simu_ref(inFile=None):
    print("*************************\n Simulation for reference state \n *************************")
    pos = [ k for k in range(CAPTVALUE) ]
    mass = [ 100 for k in range(CAPTVALUE) ]
    stiffness = [ 100 for k in range(CAPTVALUE) ]
    N = CAPTVALUE
    #simu: a activer seulement si on veut de nouvelles donnees
    #forces = simu.white_noise(100000)
    #measures = simu.simulation(pos,mass,stiffness,forces,1,"F_ref.txt","Y_ref.txt")
    
    #donnes de la simu
    forces = parser.get("F_ref.txt")[:]
    measures = parser.get("Y_ref.txt")[:]
    #
    res = []
    for i in range(10):
    	m = measures[i*10000:(i+1)*10000]
    	frf_sensor = [ [] for k in range(CAPTVALUE) ]
    	for j in range(CAPTVALUE):
    	    frf_sensor[j] = (frf.frf(forces,tool.get_lines(m,j)))
    	res.append(frf_sensor)
    if inFile == None:
    	return res
    else:
        parser.writeValues(inFile + str(i) + ".txt",res)
    	print("Values written in " + inFile + str(i) + ".txt")








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
       
        
        
#frf_multiple_capter_simu_ref("FRF_ref_")        
#frf_multiple_capter_simu("FRF_damage.txt")
#printPerCapter(frf_multiple_capter_simu(),10)
