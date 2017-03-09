import parser as pa

t = pa.get3("detect_damage_1_10percent_noise.txt")

s = 0
for x in t:
	s += x

if 0:
	print(float(s)/1800)
else:
	print(float( s - 200 ) / 1600)


