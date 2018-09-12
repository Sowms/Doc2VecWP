from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
#import tsne
#from tsne import bh_sne

File = open("schemas") #open file
schemas = File.readlines() #read all lines
File.close()

inpFile = open("avgword2vec_model","rb")
itemlist = pickle.load(inpFile)
inpFile.close()

y = np.array(itemlist)

knn = NearestNeighbors(n_neighbors=6)
#y.reshape(-1,1)
knn.fit(y)

NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=16, p=2, radius=1.0)

def calcAccuracy(neighbours, wpIn):
    k = len(neighbours[0])
    schema = schemas[wpIn]
    match = 0.0
    for i in range(1, k):
	if (schemas[neighbours[0][i]] == schema):
	    match += 1
    return match/(k-1)



avg = 0.0

for i in range(0, len(y)):
    neighbours = knn.kneighbors(np.array(y[i]).reshape(1,-1), return_distance=False)
    val = calcAccuracy(neighbours, i)
    avg = avg + val

avg = avg/len(y)
print(avg)



