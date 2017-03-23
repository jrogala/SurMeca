import parser as pa
import matplotlib.pyplot as plt

DAMAGED_SENSOR = 7

#error = [0,2,3,4,5,10,20]
#NOISE = 0.05

NOISE = 0.04
error = [5,10,15,20,25,30]


def fp(t):
	n = 0
	fp = 0
	for i in range(len(t)):
		if (i < DAMAGED_SENSOR - 2 or i > DAMAGED_SENSOR + 2):
			n += 1
			fp += t[i]
	return float(fp)/n


maxi, gauss, maxi_inv, gauss_inv = [], [], [], []

for e in error:
	t_maxi = pa.get3("detect_damage_1_" + str(e) + "percent_noise"+str(NOISE)+".txt")
	t_gauss = pa.get3("detect_damage_2_" + str(e) + "percent_noise"+str(NOISE)+".txt")
	t_maxi_inv = pa.get3("detect_damage_1_inv_" + str(e) + "percent_noise"+str(NOISE)+".txt")
	t_gauss_inv = pa.get3("detect_damage_2_inv_" + str(e) + "percent_noise"+str(NOISE)+".txt")

	maxi += [ fp(t_maxi) ]
	gauss += [ fp(t_gauss) ]
	maxi_inv += [ fp(t_maxi_inv) ]
	gauss_inv += [ fp(t_gauss_inv) ]


plt.plot(error, maxi, label = 'Maxi')
plt.plot(error, gauss, label = 'Gauss')
plt.plot(error, maxi_inv, label = 'Maxi inv')
plt.plot(error, gauss_inv, label = 'Gauss inv')
plt.xlabel("Percent of damage")
plt.ylabel("Percent of false positif")
plt.legend()
plt.show()


