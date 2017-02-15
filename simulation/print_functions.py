
import matplotlib.pyplot as plt
import numpy
import parser

###########################################################################################
###########################     Print functions    ########################################
###########################################################################################


def print_pos(tab):
	plt.plot(pos,tab)
	plt.show()

def print_minus(ref,damaged):
    n = len(ref)
    for i in range(n):
    	if (i % 10 == 0):
        	plt.plot(numpy.absolute(ref[i] - damaged[i]))
        	plt.xlabel('Sensors')
        	plt.ylabel('FRF')
        	plt.title('freq: ' + str(i))
        	plt.legend()
        	plt.show()


def printRef(ref,inf,sup):
    n = len(ref)
    delta = 3
    plt.ion()
    for i in range(n):
    	if (i%10 == 0):
        	plt.plot(numpy.absolute(ref[i]))
        	plt.plot(sup[i])
        	plt.plot(inf[i])
        	plt.pause(delta)
        	plt.clf()

def printAll(ref,inf,sup,damaged):
    n = len(ref)
    delta = 3
    #plt.ion()
    for i in range(n):
    	if (i%10 == 0):
        	plt.plot(numpy.absolute(ref[i]), label = "Reference")
        	plt.plot(numpy.absolute(sup[i]))
        	plt.plot(numpy.absolute(inf[i]))
        	plt.plot(numpy.absolute(damaged[i]), label = "Damaged")
        	plt.xlabel('Sensors')
        	plt.ylabel('FRF')
        	plt.title('freq: ' + str(i))
        	plt.legend()
        	#plt.pause(delta)
        	#plt.clf()
        	plt.show()

def printInterpolationError(ref,inf,sup,undamaged,damaged):
    n = len(ref)
    delta = 3
    #plt.ion()
    for i in range(n):
    	if (i % 10 == 0):
			undam = interpolation.interpolation_error(undamaged[i])
			dam = interpolation.interpolation_error(damaged[i])
			plt.plot(numpy.absolute(undam), label = "Undamaged")
			plt.plot(numpy.absolute(dam), label = "Damaged")
			plt.xlabel('Sensors')
			plt.ylabel('Error')
			plt.title('freq: ' + str(i))
			plt.legend()
        	#plt.pause(delta)
        	#plt.clf()
			plt.show()


def printAllMoustache(refs,undamaged, damaged):
    n = len(undamaged)
    delta = 3
    #plt.ion()
    for i in range(n):
        if (i % 10 == 0):
            l = [[] for j in range(len(refs[0]))]
            for exp in range(len(refs)):
                for captor in range(len(refs[exp])):
                    l[captor] += [refs[exp][captor][i]]
            l = [numpy.absolute(d) for d in l]
            plt.boxplot(l,positions = [i for i in range(len(l))])
            plt.plot(numpy.absolute(undamaged[i]), label = "Undamaged")
            plt.plot(numpy.absolute(damaged[i]), label = "Damaged")
            plt.xlabel('Sensors')
            plt.ylabel('FRF')
            plt.title('freq: ' + str(i))
            plt.legend()
            #plt.pause(delta)
            #plt.clf()
            plt.show()
