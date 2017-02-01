def get_lines(t,j):
    s = []
    for i in range(len(t)):
        try:
            s.append(t[i][j])
        except:
            continue
    return s
