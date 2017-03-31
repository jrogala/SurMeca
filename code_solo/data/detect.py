import parser as pa
import matplotlib.pyplot as plt

error = [0,1, 2, 3, 4, 5, 8, 10, 13, 15, 20, 30,40,50,60,70]#12
error = [1,5,10,15,20,30,40,50,60,70]#il faudrait ajouter autour de 18

DAMAGED_SENSOR = 7

maxi, gauss, maxi_inv, gauss_inv = [], [], [], []

for e in error:
	t_maxi = pa.get3("detect_damage_1_" + str(e) + "percent.txt")
	t_gauss = pa.get3("detect_damage_2_" + str(e) + "percent.txt")
	t_maxi_inv = pa.get3("detect_damage_1_inv_" + str(e) + "percent.txt")
	t_gauss_inv = pa.get3("detect_damage_2_inv_" + str(e) + "percent.txt")

	maxi += [ t_maxi[DAMAGED_SENSOR] ]
	gauss += [ t_gauss[DAMAGED_SENSOR] ]
	maxi_inv += [ t_maxi_inv[DAMAGED_SENSOR] ]
	gauss_inv += [ t_gauss_inv[DAMAGED_SENSOR] ]


plt.plot(error, maxi, label = 'Maxi')
plt.plot(error, gauss, label = 'Gauss')
plt.plot(error, maxi_inv, label = 'Maxi inv')
plt.plot(error, gauss_inv, label = 'Gauss inv')
plt.xlabel("Percent of damage")
plt.ylabel("Percent of localization")
plt.legend()
plt.show()


