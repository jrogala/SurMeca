import parser as pa
import matplotlib.pyplot as plt

noise = [0.05,0.1,0.2]

DAMAGED_SENSOR = 7
#NOISE = 0.05
DAMAGE = 10#percent of damage

def tp(t):
	return max(t[DAMAGED_SENSOR-1], t[DAMAGED_SENSOR])


maxi, gauss, maxi_inv, gauss_inv = [], [], [], []

for n in noise:
	t_maxi = pa.get3("detect_damage_1_" + str(DAMAGE) + "percent_noise"+str(n)+".txt")
	t_gauss = pa.get3("detect_damage_2_" + str(DAMAGE) + "percent_noise"+str(n)+".txt")
	t_maxi_inv = pa.get3("detect_damage_1_inv_" + str(DAMAGE) + "percent_noise"+str(n)+".txt")
	t_gauss_inv = pa.get3("detect_damage_2_inv_" + str(DAMAGE) + "percent_noise"+str(n)+".txt")

	maxi += [ tp(t_maxi) ]
	gauss += [ tp(t_gauss) ]
	maxi_inv += [ tp(t_maxi_inv) ]
	gauss_inv += [ tp(t_gauss_inv) ]


plt.plot(noise, maxi, label = 'Maxi')
plt.plot(noise, gauss, label = 'Gauss')
plt.plot(noise, maxi_inv, label = 'Maxi inv')
plt.plot(noise, gauss_inv, label = 'Gauss inv')
plt.xlabel("Percent of noise")
plt.ylabel("Percent of localization")
plt.title("Damage: " + str(DAMAGE) + "%")
plt.legend()
plt.show()


