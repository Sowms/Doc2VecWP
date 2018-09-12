import nltk
from nltk.tokenize import RegexpTokenizer

quesfile = open("questions","r")
data = quesfile.readlines()
quesfile.close()

dictfile = open("dictionary", "w")
dictionary = []

for i in range(0, len(data)):
    text = data[i]
    text = text.lower()
    doc = nltk.tokenize.word_tokenize(text)
    for word in doc:
	if (word not in dictionary):
	    dictionary.append(word)
            dictfile.write(word+"\n")   

dictfile.close()
