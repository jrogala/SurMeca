import core

f = core.Simulation()
f.experiment([100,80,100,100,100,100,100,100,100,100],[100 for k in range(10)],10000)
f.save("10mass10000timesDamaged1.txt")
