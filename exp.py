import main
import interpolation
import matplotlib.pyplot as plt
import numpy
import parser


###########################################################################################
###########################     Print functions    ########################################
###########################################################################################


def printRef(ref,inf,sup):
    n = len(ref)
    delta = 3
    plt.ion()
    for i in range(n):
    	if (i%50 == 0):
        	plt.plot(numpy.absolute(ref[i]))
        	plt.plot(numpy.absolute(sup[i]))
        	plt.plot(numpy.absolute(inf[i]))
        	plt.pause(delta)
        	plt.clf()

def printAll(ref,inf,sup,undamaged, damaged):
    n = len(ref)
    delta = 3
    plt.ion()
    for i in range(n):
    	if ((i%20 == 0) and (i>300)):
        	plt.plot(numpy.absolute(ref[i]), label = "Reference")
        	plt.plot(numpy.absolute(sup[i]))
        	plt.plot(numpy.absolute(inf[i]))
        	plt.plot(numpy.absolute(undamaged[i]), label = "Undamaged")
        	plt.plot(numpy.absolute(damaged[i]), label = "Damaged")
        	plt.xlabel('Sensors')
        	plt.ylabel('FRF')
        	plt.legend()
        	plt.pause(delta)
        	plt.clf()




###########################################################################################
###########################     Experience    #############################################
###########################################################################################



def freq_sampl(fichier_sampl):
	f = open(fichier_sampl,'r')
	freq = float(f.readline())
	return freq


def experience():
	#frf_list_ref = main.frf_multiple_capter_simu_ref()
	frf_list_ref = [ [] for i in range(main.MAKEREF) ]
	for i in range(main.MAKEREF):
		frf_list_ref[i] = parser.get("FRF_ref" + str(i) + ".txt")
	
	
	
	frf_undamaged = numpy.transpose(main.frf_multiple_capter_simu_undamaged("FRF_undamaged.txt"))
	frf_undamaged = numpy.transpose(parser.get("FRF_undamaged.txt"))
	
	
	frf_damaged = numpy.transpose(main.frf_multiple_capter_simu_damaged("FRF_damaged.txt"))
	frf_damaged = numpy.transpose(parser.get("FRF_damaged.txt"))
	
	
	
	#f_sampl_ref = freq_sampl("f_sampl_ref.txt")
	#f_sampl_undamaged = freq_sampl("f_sampl_undamaged.txt")
	#f_sampl_damaged = freq_sampl("f_sampl_damaged.txt")


	print("*************************\n Average frf calcul \n *************************")

	nb_frf = len(frf_list_ref)
	nb_sensor = len(frf_list_ref[0])
	nb_freq = len(frf_list_ref[0][0])

	print("Nombre de frf par capteur et par frequence:" + str(nb_frf))
	print("Nombre de capteur:" + str(nb_sensor))
	print("Nombre de frequences:" + str(nb_freq))
	
	#print("Frequence d'echantillonage reference: " + str( f_sampl_ref ) )
	#print("Frequence d'echantillonage (undamaged): " + str( f_sampl_undamaged ) )
	#print("Frequence d'echantillonage (damaged): " + str( f_sampl_damaged ) )
	
	
	ref, inf, sup = interpolation.create_ref( frf_list_ref )
	
	#printRef(ref,inf,sup)
	
	error_ref = interpolation.global_error(ref)
	error_undamaged = interpolation.global_error(frf_undamaged)
	error_damaged = interpolation.global_error(frf_damaged)
	
	
	undamaged = [0 for i in range(nb_sensor) ]
	damaged = [0 for i in range(nb_sensor) ]
	for i in range(nb_sensor):
		undamaged[i] = error_undamaged[i] - error_ref[i]
		damaged[i] = error_damaged[i] - error_ref[i]
	
	print("*************************\n Errors \n *************************")
	print("Error ref: ")
	print(error_ref)
	print("Undamaged state: ")
	print(undamaged)
	print("Damaged state: ")
	print(damaged)
	
	printAll(ref,inf,sup, frf_undamaged, frf_damaged)




###########################################################################################
###########################     Script    #################################################
###########################################################################################




experience()
