#!/usr/bin/python

# Silly Putty clustering algorithm

# This implementation: Alex Rasmussen, rasmussen.99@osu.edu
# Original design: Kevin Coombes, OSU


import numpy, random, math
from scipy import spatial


# new_row = (x - x[:,j])^2
# x[:,j]
# matrix[j,:] = new_row
# matrix[:,j] = new_row



#
# the silly putty object
#

# the high-speed kicker is to use built-in vector operations
#
# in much of this class, we use the index for a array or list as the identifier for the data
# or cluster at hand.  thus, they should never be sorted or reordered, lest the cluster information 
# be out of sync with the data points


class sillyobject:
    # Constructor
    #
    # set up the variables (including the temp data list)
    def __init__(self, metric, cluster_number, loop_length, max_steps):
    
        self.metric = metric
        self.cluster_number = cluster_number
        self.loop_length = loop_length
        self.max_steps = max_steps
        
        self.data_list = []
        
        self.cluster_count = numpy.array([0 for x in range(self.cluster_number)])
        
        self.last_moved_points = []
        
    
    # add points to the initial data list
    def add_point(self, inputpoint):
        self.data_list.append(inputpoint)
    
    
    # startup creates the array, creates the aux info arrays, and computes the cluster count
    #
    # we also get the first pass at the distance matrix here as well
    #
    # not in constructor because growing numpy arrays is expensive
    
    # functions to add
    # def randomize -- resets the cluster assignments
    #
    # def change_k -- (re)builds the cluster arrays
    #
    # def cluster -- returns cluster assignment array
    
    
    
    def startup(self):
    
        self.steps = 0
    
        # get the number of data points
        self.N = len(self.data_list)
        
        # init the main data array
        self.data_array = numpy.array(self.data_list)
               
        # init the cluster array, assigning random clusters to the data points
        self.cluster_array = numpy.array([[0.0 for x in range(self.cluster_number)] for y in range(self.N)])

        for i in range(self.N):
            self.cluster_array[i,random.randrange(self.cluster_number)] = 1.0
        
        # init the silhouette array, setting them all to zero
        self.silhouette_array = numpy.array([0.0 for x in range(self.N)])
        
        # init the auxilliary cluster array, setting the first to the (already assigned) cluster
        # and the nearest cluster to zero
        self.cluster_aux_array = numpy.array([[numpy.argmax(self.cluster_array[x,:]),0] for x in range(self.N)])
        
        # count the clusters, noting that the position in the cluster list
        # is the index for the clusters 
        self.cluster_count = numpy.sum(self.cluster_array, axis=0)
        
        # create the distance array for quickly calculating distances
        self.distance_array = spatial.distance.cdist(self.data_array, self.data_array, self.metric)
                    
               
    
    # the silouhette updater, which is much faster due to the distances being precalculated
    #
    # here again we have to be careful that the row index on the array is used to both index the data
    # and cluster arrays
    #@profile
    def move_worst_point(self):
    
        normed_distances = numpy.array([0.0 for x in range(self.cluster_number)])
        
        own_dist = 0.0
        own_cluster = 0
        nearest_dist = 0.0
        nearest_cluster = 0
    
        # first, multiply the master cluster distance matrices
        #
        # this yields a (N x cluster_number) array of sum of distances to each cluster (columns)
        # for each point (row)
        cluster_distances = self.distance_array @ self.cluster_array

        # for each point (i.e row in cluster_distances), we normalize the distances
        #
        # in principle this could be a matrix op too, but the vector of matrices depends on the 
        # size of each cluster and would have to be updated at each step
        for i in range(self.N):
        
            # normalize the distance sums, ignoring the -1 from the own cluster
            normed_distances = numpy.divide(cluster_distances[i,:], self.cluster_count)
        
            # pull out the distance to the current cluster
            own_cluster = self.cluster_aux_array[i,0]
            own_dist = normed_distances[own_cluster]
            
            # replace that point with infinity, then find the closest cluster  
            normed_distances[own_cluster] = numpy.inf
            nearest_cluster = numpy.argmin(normed_distances)
            nearest_dist = normed_distances[nearest_cluster]
            
            # the nearest cluster needs to be saved to move the point
            self.cluster_aux_array[i,1] = nearest_cluster
            
            # compute the silhouette width
            #
            # since we drop the -1 from own cluster, a cluster with one datapoint will be assigned 0
            self.silhouette_array[i] = (nearest_dist - own_dist)/max([nearest_dist, own_dist])
            
                 
        # find the worst point
        worst_point = numpy.argmin(self.silhouette_array)
    
        # if the silhouette width is negative, move the worst point to its nearest cluster
        # and update the counts
        #
        # in principle, we could also set a cutoff here as well
        if self.silhouette_array[worst_point] < 0:
        
            # update the cluster counts
            #print('moving point %s from %s to %s' % (worst_point,numpy.argmax(self.cluster_array[i,:]),self.nearest_cluster_array[i]))
            
            self.cluster_count[self.cluster_aux_array[worst_point,0]] -= 1.0
            self.cluster_count[self.cluster_aux_array[worst_point,1]] += 1.0
            
            # reassign the cluster
            #self.cluster_array[i,:] = numpy.array([1.0 if x == self.nearest_cluster_array[i] else 0.0 for x in range(self.cluster_number)])

            self.cluster_array[worst_point,self.cluster_aux_array[worst_point,0]] = 0.0
            self.cluster_array[worst_point,self.cluster_aux_array[worst_point,1]] = 1.0
        
            self.cluster_aux_array[worst_point,0] = self.cluster_aux_array[worst_point,1]

            
        # stopping condition will be handled by a separate function, which looks at the list of 
        # recently moved points, and should be called after each silhouette_update
        #
        # add the point to the stopping list, then slice to the specified length
        #
        # if we wanted to keep track of the old and new clusters for this point, include that here
        self.last_moved_points.append([worst_point, self.silhouette_array[worst_point]])
        self.last_moved_points = self.last_moved_points[-self.loop_length:]
        
        # finally, increment the step counter
        self.steps += 1
        

    # function that determines if the algorithm should stop
    def stop(self):
    
        # make sure the algorithm has run once
        if self.steps == 0:
            return False
        
        # first, stop if no negative silhouette widths remain to be moved
        if [x[1] for x in self.last_moved_points if x[1] < 0] == []:
            print('all silhouette widths positive')
            return True
            
        # second, check if max_steps has been reached
        elif self.steps >= self.max_steps:
            print('max steps reached')
            return True
            
        # third, check if the moved points are caught in a loop
        #elif islooping(list):
        #    return True
            
        else:
            return False
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
