import sys
import random

datafile = sys.argv[1]
f = open(datafile)
#f =open("/Users/pavanghuge/Downloads/ML/Assignment8/ionosphere.data")
data = []
i = 0
l = f.readline()

# Read Datafile
while (l != ''):
	a = l.split()
	l2 = []
	for j in range(0, len(a), 1):
		l2.append(float(a[j]))
	data.append(l2)
	l = f.readline()
rows = len(data)
cols = len(data[0])
f.close()

#k = int(sys.argv[2])
k = int(input("Enter the number of clusters as an integer value =  "))

#This function assigns the data randomly into clusters
def classes(data, k):    
    
    clusters = {}  
    for i in range(0, k):
        clusters[i] = []
    
    for val in data:
        i = random.randint(0, k-1)
        clusters[i].append(val)
        
    return clusters

#This function returns the Euclidean Distance between two given points
def distance(P, Q):
    dist = 0
    for a, b in zip(P, Q):
        dist += (a - b) ** 2
        
    dist = dist ** 0.5
    return dist

#This function returns the centroids of all the points in the given cluster.
def centroid(clusters):
    centroids = {}
    for C in clusters.keys():
        cluster_T = list(zip(*clusters[C]))
        centroid_point = []
        for colm in cluster_T:
            mean = sum(colm)/len(colm)
            centroid_point.append(mean)
            centroids[C] = centroid_point
        
    return centroids

#This function reallocates the data into clusters.
def clusters(data, centroids):
    clusters = {}
    for key in centroids.keys():
        clusters[key] = []
    
    for val in data:
        dist_list = []
            
        for k, centroid in centroids.items():
            dist = distance(val, centroid)
            dist_list.append(dist)      
        min_dist = min(dist_list)
        cluster_num = dist_list.index(min_dist)
        clusters[cluster_num].append(val)
        
    return clusters 

#This function returns the objective value for given cluster.
def objective(clusters, centroids):
    obj =  0   
    for C in clusters.keys():
        for point in clusters[C]:
            for val, cent in zip(point, centroids[C]):
                obj += (val - cent) ** 2    
    return obj

# K-Means algorithm and returns clustered data.
def k_means(data, k):
    cluster = classes(data, k)
    centroids = centroid(cluster)
    old_obj = objective(cluster, centroids)
        
    while True:
        new_clusters = clusters(data, centroids)
        new_centroids = centroid(new_clusters)
        new_obj = objective(new_clusters, new_centroids)
            
        if old_obj == new_obj:
            break
        old_obj = new_obj
    
    return new_clusters
 
#This function is for labeling the clustered data returns labels and index
def labeling(data, clusters):
    label = {}
        
    for lab, points in clusters.items():
        for point in points:
            i = data.index(point)
            label[i] = lab
    return label

# Calling the finctions for clustering and labeling
clustered = k_means(data, k)
labels = labeling(data, clustered)

for i in sorted(labels.keys()):
    print(labels[i], i)