import nltk
from nltk.tokenize import RegexpTokenizer

File = open("illinois-questions") #open file
data = File.readlines() #read all lines
File.close()

f1 = open("illinois-output","r")
output = f1.readlines();
f1.close()

newfile = open("annotated-illinois-questions","w")

cin = ['get','collect','find','takes','eat','ate','found','finds','collects']
cout = ['give','put','gave','gives','puts','loses']
cplus = ['heavier']
cminus = []
grp = ['total', 'altogether','left']
vry = ['each', 'equally','per']

for i in range(0, len(data)):
    sentences = nltk.sent_tokenize(data[i]) #tokenize sentences
    newline = ""
    cont = []
    ent = []
    quesflag = False
    for sentence in sentences:
        for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
	    if (word.lower() in cin):
		newline = newline + "SCH:CIN "
            if (word.lower() in cout):
		newline = newline + "SCH:COUT "
            if (word.lower() in cplus):
		newline = newline + "SCH:CPLUS "
            if (word.lower() in cminus):
		newline = newline + "SCH:CMINUS "
	    if (word.lower() in grp):
		newline = newline + "SCH:GRP "
            if (word.lower() in vry):
		newline = newline + "SCH:VRY "
        
	    if (word.lower() == "how"):
		quesflag = True
            if (pos == 'NNP'):
                if (word not in cont):
                    cont.append(word)
                    if (quesflag):
		        newline = newline + "QUESCONT "
		    else:
			newline = newline + "CONT" + str(len(cont)) + " "
                else:
                    num = cont.index(word) + 1
                    if (quesflag):
		        newline = newline + "QUESCONT "
		    else:
			newline = newline + "CONT" + str(num) + " "

	    if (pos == 'NNS' or pos == 'NNPS'):
                if (word not in ent):
                    ent.append(word)
		    if (quesflag):
		        newline = newline + "QUESENT "
		    else:
			newline = newline + "ENT" + str(len(ent)) + " "
                else:
                    if (quesflag):
		        newline = newline + "QUESENT "
		    else:
			num = ent.index(word) + 1
		        newline = newline + "ENT" + str(num) + " "
            newline = newline + word + " "
    newline = newline.strip()
    newfile.write(newline+"\n")


