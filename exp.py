import main
import interpolation
import matplotlib.pyplot as plt
import numpy



def printRef(ref,inf,sup):
    n = len(ref)
    delta = 1
    plt.ion()
    for i in range(n):
    	if (i%100 == 0):
        	plt.plot(numpy.absolute(ref[i]))
        	plt.plot(numpy.absolute(sup[i]))
        	plt.plot(numpy.absolute(inf[i]))
        	plt.pause(delta)
        	plt.clf()



def experience():
	frf_list_ref = main.frf_multiple_capter_simu_ref()
	frf_undamaged = main.frf_multiple_capter_simu_undamaged()
	frf_damaged = main.frf_multiple_capter_simu_damaged()


	print("*************************\n Average frf calcul \n *************************")

	nb_frf = len(frf_list_ref)
	nb_sensor = len(frf_list_ref[0])
	nb_freq = len(frf_list_ref[0][0])

	print("Nombre de frf par capteur et par frequence:" + str(nb_frf))
	print("Nombre de capteur:" + str(nb_sensor))
	print("Nombre de frequences:" + str(nb_freq))
	
	ref, inf, sup = interpolation.create_ref( frf_list_ref )
	
	printRef(ref,inf,sup)
	
	
	
	







experience()
