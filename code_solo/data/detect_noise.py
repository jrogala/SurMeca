import parser as pa
import matplotlib.pyplot as plt


DAMAGED_SENSOR = 7

#NOISE = 0.05
#error = [0,2,3,4,5,6,8,10,20]

#NOISE = 0.1
#error = [2,3,4,5,7,8,9,10,12,14,15,20]

NOISE = 0.04
error = [5,10,15,20,25,30,40]

def tp(t):
	return max(t[DAMAGED_SENSOR-1], t[DAMAGED_SENSOR])


maxi, gauss, maxi_inv, gauss_inv = [], [], [], []

for e in error:
	t_maxi = pa.get3("detect_damage_1_" + str(e) + "percent_noise"+str(NOISE)+".txt")
	t_gauss = pa.get3("detect_damage_2_" + str(e) + "percent_noise"+str(NOISE)+".txt")
	t_maxi_inv = pa.get3("detect_damage_1_inv_" + str(e) + "percent_noise"+str(NOISE)+".txt")
	t_gauss_inv = pa.get3("detect_damage_2_inv_" + str(e) + "percent_noise"+str(NOISE)+".txt")

	maxi += [ tp(t_maxi) ]
	gauss += [ tp(t_gauss) ]
	maxi_inv += [ tp(t_maxi_inv) ]
	gauss_inv += [ tp(t_gauss_inv) ]


plt.plot(error, maxi, label = 'Maxi')
plt.plot(error, gauss, label = 'Gauss')
plt.plot(error, maxi_inv, label = 'Maxi inv')
plt.plot(error, gauss_inv, label = 'Gauss inv')
plt.xlabel("Percent of damage")
plt.ylabel("Percent of localization")
plt.title("Noise: " + str(NOISE))
plt.legend()
plt.show()


