from sklearn.cluster import KMeans
import pickle
import numpy as np

f1 = open("schemas-cl","r")
output = f1.readlines();
f1.close()

intoutput = []
for i in output:
    i = i.rstrip();
    intoutput.append(int(i))

inpFile = open("autoenc_model","rb")
itemlist = pickle.load(inpFile)
inpFile.close()

encoded_arr = np.array(itemlist)

avg = 0.0
arrY = encoded_arr

for j in range(0, 100):
    kmeans_model = KMeans(n_clusters=4, init='k-means++', max_iter=100)  
    X = kmeans_model.fit(arrY)
    labels = kmeans_model.labels_.tolist()
    l = kmeans_model.fit_predict(arrY)
    #computing purity
    #3 clusters in l and 3 classes in output 
    #Step 1 - identifying N
    N = len(intoutput)
    #Step 2 - aggregating each cluster
    matrix = [[0 for x in range(4)] for m in range(4)] 
    matrix = np.zeros_like(matrix)
    for i in range(0, len(labels)):
    #print(`labels[i]` + "|" + `intoutput[i]`)
        matrix[labels[i]][intoutput[i]] += 1
    num = 0
    for i in range(0, 4):
        num += max(matrix[i])
    purity = num/float(N)
    avg = avg + purity
avg = avg/100
print(avg)
