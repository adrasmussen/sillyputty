#!/usr/bin/python

import sillyputty, math, code, numpy
from matplotlib import pyplot

import cProfile

from sklearn.datasets.samples_generator import make_blobs

gen_clusters = 5
sp_clusters = 5
N = 500
max_steps = 3000

runs = 50

gen_array, gen_cluster = make_blobs(n_samples=N, centers=gen_clusters, n_features=2, random_state=0, cluster_std=0.1)




gen_plot = pyplot.figure(1)
pyplot.scatter(gen_array[:,0], gen_array[:,1], c=gen_cluster)
pyplot.title('generated data')

sillytest = sillyputty.sillyobject('euclidean', sp_clusters, 10, max_steps)

# add data
for i in range(N):
    sillytest.add_point(gen_array[i,:])



# run the init system
sillytest.startup()

output_list = []


for i in range(runs):
    sillytest.randomize_clusters()
    sillytest.cluster_points()

    c = sillytest.cluster_array @ sillytest.cluster_array.transpose() - numpy.identity(N)
    output_list.append(c)
    
output_array = numpy.array(output_list)

comp_array = numpy.sum(output_array, axis=0)

    

#code.interact(local=locals())  



sp_array = sillytest.data_array
sp_cluster = sillytest.cluster_aux_array[:,0]


gen_plot = pyplot.figure(2)
pyplot.scatter(sp_array[:,0], sp_array[:,1], c=sp_cluster)
pyplot.title('clustered data')


gen_plot = pyplot.figure(3)
pyplot.hist(comp_array.flatten(), bins=runs)


pyplot.show()

#code.interact(local=locals())
