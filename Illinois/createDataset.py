import nltk
import json



with open('illinois/questions.json') as f1:
    data = json.load(f1)
f1.close()

lines = []
operators = []

# -*- encoding : utf-8 -*-

for i in range(0, len(data)):
    line = data[i]["sQuestion"]
    lines.append(line)

# 0 - +, 1 - -, 2 - *, 3 - /
for i in range(0, len(data)):
    equation = data[i]["lEquations"][0]
    operator = "-1"
    if ('+' in equation):
        operator = "0"
    elif ('-' in equation):
	operator = "1"
    elif ('*' in equation):
	operator = "2"
    elif ('/' in equation):
	operator = "3"
    operators.append(operator)

newfile = open("illinois-output","w")
counter = 0

for line in operators:
    newfile.write(line+"\n")

newfile.close()

newfile = open("illinois-questions","w")

for line in lines:
    newfile.write(line+"\n")

newfile.close()
