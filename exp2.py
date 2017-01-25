import main
import interpolation as itp
import matplotlib.pyplot as plt
import numpy
import parser
import simu
import print_functions as pr

###########################################################################################
###########################     Variables    ##############################################
###########################################################################################


CAPTVALUE = simu.CAPTVALUE
SAMPLVALUE = simu.SAMPLVALUE
MAKEREF = simu.MAKEREF
DAMAGED_SENSOR = simu.DAMAGED_SENSOR
DAMAGE = simu.DAMAGE

indic = ""
#indic = "stiffness_random"


###########################################################################################
###########################     Experience    #############################################
###########################################################################################

def experience():
	main.frf_multiple_sensors("data/FRF_ref"+indic+".txt","data/FRF_inf"+indic+".txt","data/FRF_sup"+indic+".txt","data/FRF_damaged"+indic+".txt",remake=False)
	ref = parser.get2("data/FRF_ref"+indic+".txt")
	inf = parser.get2("data/FRF_inf"+indic+".txt")
	sup = parser.get2("data/FRF_sup"+indic+".txt")
	frf_damaged = numpy.transpose(parser.get2("data/FRF_damaged"+indic+".txt"))

	#pr.printRef(ref,inf,sup)
	pr.printAll(ref,inf,sup, frf_damaged)
	#e = itp.if_out(inf,sup,frf_damaged)
	#pr.print_pos(e)
	#pr.print_minus(ref,frf_damaged)


###########################################################################################
###########################     Script    #################################################
###########################################################################################




experience()
