import sys
f = open(sys.argv[1], "r")
lines = [line[:-1] for line in f.readlines()]

q = {}
for i in range(70):
    query = lines[i].split(",")[0]
    q[query] = 1

o = open("sub.txt", "w")
o.write("id,images\n")
for i in range(70):
    parts = lines[i].split(",")
    query = parts[0]
    o.write(query + ",")
    candidatesAndScores = parts[1].split(" ")
    for c in range(0, len(candidatesAndScores), 2):
        if candidatesAndScores[c] in q:
            continue
        o.write(candidatesAndScores[c] + " ")
    o.write("\n")
    o.flush()

o.close()

