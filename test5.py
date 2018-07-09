import nltk
import json

'''
File = open("q2.txt") #open file
lines = File.readlines() #read all lines
File.close()
'''

with open('AddSub.json') as f1:
    dataAddSub = json.load(f1)
f1.close()

lines = []

for i in range(0, len(dataAddSub)):
    lines.append(dataAddSub[i]["sQuestion"])

with open('SingleOp.json') as f2:
    dataSingleOp = json.load(f2)
f2.close()

for i in range(0, len(dataSingleOp)):
    lines.append(dataSingleOp[i]["sQuestion"])

newfile = open("entire-num-prop-ent","w")
counter = 0

for line in lines:
    sentences = nltk.sent_tokenize(line) #tokenize sentences
    newline = lines[counter]
    for sentence in sentences:
	print sentence
        for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
            if (pos == 'CD'):
                newline = newline.replace(word, "NUMTOK")
            if (pos == 'NNP' or pos == 'NNPS'):
                newline = newline.replace(word, "CONTTOK")
	    if (pos == 'NNS'):
                newline = newline.replace(word, "ENTTOK")
    counter += 1
    newfile.write(newline)

newfile.close()

