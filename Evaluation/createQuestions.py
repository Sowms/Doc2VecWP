import json

quesfile = open("questions","w")
ansfile = open("answers", "w")

with open('SingleOp.json') as f2:
    dataSingleOp = json.load(f2)
f2.close()

for i in range(0, len(dataSingleOp)):
    quesfile.write(dataSingleOp[i]["sQuestion"])
    quesfile.write('\n')
    ansfile.write(str(dataSingleOp[i]["lSolutions"][0]))
    ansfile.write('\n')

quesfile.close()
ansfile.close()
