import cmath

def get(path):
    s = []
    with open(path,"r") as f:
        for line in f:
            values = []
            for value in line.split(" "):
                try:
                    values.append(float(value))
                except:
                    continue
            if len(values) == 1:
                s.append(values[0])
            else:
                s.append(values)
    return s

def writeValues(path,res):
    with open(path,"w") as f:
        for line in res:
            s = ""
            for value in line:
                if s == "":
                    s += "" + str(value)
                else:
                    s += " " + str(value)
            f.write(s + "\n")



def get2(path):
    s = []
    with open(path,"r") as f:
        for line in f:
            values = []
            for value in line.split(" "):
                try:
                    values.append(complex(value))
                except:
                    continue
            if len(values) == 1:
                s.append(values[0])
            else:
                s.append(values)
    #try:
    #    print(path + " done. Xsize= " + str(len(s)) + " Ysize= " + str(len(s[0])))
    #except:
    #    print(path + " done. Xsize= " + str(len(s)))
    return s











