import numpy as np
#from skdata.mnist.views import OfficialImageClassification
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
import pickle
#from tsne import bh_sne
# load up data
File = open("schemas-cl") #open file
schemas = File.readlines() #read all lines
File.close()


inpFile = open("siamese_model_perturb_onlyNP","rb")
itemlist = pickle.load(inpFile)
inpFile.close()

x_data = np.asarray(itemlist)
y_data = schemas
# convert image data to float64 matrix. float64 is need for bh_sne
x_data = np.asarray(x_data).astype('float64')
x_data = x_data.reshape((x_data.shape[0], -1))
# For speed of computation, only run on a subset
n = 562
x_data = x_data[:n]
y_data = y_data[:n]
# perform t-SNE embedding
vis_data = TSNE(learning_rate=100).fit_transform(x_data)
# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

label1 = ["#FFFF00", "#008000", "#0000FF", "#FF0000"]
col = [label1[int(i)] for i in schemas]

#markers = ["s","o","^","p"]
#for i, c in enumerate(np.unique(col)):
 #   plt.scatter(vis_data[:,0][col==c],vis_data[:,1][col==c],c=col[col==c], marker=markers[i])
plt.scatter(vis_x, vis_y, c=col)
#plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()
