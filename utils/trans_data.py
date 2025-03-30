

def trans(data):
    line = data.split("\n")
    out = []
    for l in line:
        if len(v := l.split('\t')) > 1:
            out.append([float(i) for i in v])
    print(out)


if __name__ == '__main__':
    data = """
1.00 	1.00 	1.00 	0.00 	0.00 	0.41 
1.00 	1.00 	1.00 	0.00 	0.00 	0.41 
1.00 	1.00 	1.00 	0.00 	0.00 	0.19 
0.00 	0.00 	0.00 	0.00 	0.00 	0.00 
0.00 	0.00 	0.00 	0.00 	0.00 	0.00 

    """

    trans(data)