import parser
import matplotlib.pyplot as plt


DAMAGED_SENSOR = 7

#NOISE = 0.05
#error = [0,2,3,4,5,6,8,10,20]

NOISE = 0.1
damage = [0,5,10,20,30,40,50,60,70,80,90,100]

NOISE = 0

q1, q2, q1_inv, q2_inv = [], [], [], []


for DAMAGE in damage:
	d1 = parser.get3("quantification1_"+str(DAMAGE)+"damage_"+str(NOISE)+"noise.txt")
	d2 = parser.get3("quantification2_"+str(DAMAGE)+"damage_"+str(NOISE)+"noise.txt")
	d1_inv = parser.get3("quantification1_inv_"+str(DAMAGE)+"damage_"+str(NOISE)+"noise.txt")
	d2_inv = parser.get3("quantification2_inv_"+str(DAMAGE)+"damage_"+str(NOISE)+"noise.txt")
	q1 += [max(d1[DAMAGED_SENSOR],d1[DAMAGED_SENSOR-1])]
	q2 += [max(d2[DAMAGED_SENSOR],d2[DAMAGED_SENSOR-1])]
	q1_inv += [max(d1_inv[DAMAGED_SENSOR],d1_inv[DAMAGED_SENSOR-1])]
	q2_inv += [max(d2_inv[DAMAGED_SENSOR],d2_inv[DAMAGED_SENSOR-1])]

plt.plot(damage, q1, label = 'Maxi')
plt.plot(damage, q2, label = 'Gauss')
plt.plot(damage, q1_inv, label = 'Maxi inv')
plt.plot(damage, q2_inv, label = 'Gauss inv')
plt.xlabel("Percent of damage")
plt.ylabel("difference")
plt.title("Quantification")
plt.legend()
plt.show()
