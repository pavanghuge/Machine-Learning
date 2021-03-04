import sys
import random

#datafile = sys.argv[1]
#f = open(datafile)
f =open("/Users/pavanghuge/Downloads/ML/Assignment8/ionosphere.data")
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

try:
	k = int(input("Enter the number of clusters: "))
except IndexError:
	print(" Improper Syntax: try - python3 Assignment8_Pavan_Ghuge.py datafile 2 {value of k>1}")
	sys.exit()

#initialize coordinates in means
    # number of dimensions is equal to number of coordinates of mean
    
mean = []
col = []
for j in range(0, cols, 1):
	col.append(0)

# k cluster will have k means
for i in range(0, k, 1):
	mean.append(col)

# Initially dividing dataset into k clusters randomnly  
random1 = 0
for p in range(0, k, 1):
	random1=random.randrange(0,(rows-1))
	mean[p] = data[random1]

#classifying points

cluster = {}
diff = 1

prev = [[0]*cols for x in range(k)]

dist, n , mean_dist =[],[],[]

for p in range(0, k, 1):
	mean_dist.append(0)
	dist.append(0.1)
	n.append(0.1)


total_dist =1
classes=[]

while ((total_dist) > 0):
	for i in range(0,rows, 1):
		dist =[]

		for p in range(0, k, 1):
			dist.append(0)
		for p in range(0, k, 1):
			for j in range(0, cols, 1):
				dist[p] += ((data[i][j] - mean[p][j])**2)
		for p in range(0, k, 1):
			dist[p] = (dist[p])**0.5
		minimum_dist = 0
		minimum_dist = min(dist)
                
		for p in range(0, k, 1):
			if(dist[p]==minimum_dist):
				cluster[i] = p                               
				n[p]+=1                
				break
               
    # compute means

	mean = [[0]*cols for x in range(k)]
	col = []    

	for i in range(0, rows, 1):
		for p in range(0, k, 1):
            
			if(cluster.get(i) == p):
				for j in range(0, cols, 1):                    
					temp =  mean[p][j]
					temp1 =  data[i][j]
					mean[p][j] = temp + temp1                    
                    
	for j in range(0, cols, 1):
		for i in range(0, k, 1):
			mean[i][j] = mean[i][j]/n[i]

	classes = [int(x) for x in n]
	n=[0.1]*k    
    
    #compute distance

	mean_dist = []
	for p in range(0, k, 1):
		mean_dist.append(0)
	for p in range(0, k, 1):
		for j in range(0, cols, 1):
			mean_dist[p]+=float((prev[p][j]-mean[p][j])**2)

		mean_dist[p] = (mean_dist[p])**0.5
    
	prev=mean
	total_dist = 0
	for b in range(0,len(mean_dist),1):
		total_dist += mean_dist[b]

#	print ("the distance between means:",totaldist)
print(" Number of Data points in",k," clusters are",classes)

#clustering of unclustered data
#cluster
for i in range(0,rows, 1):
	print(cluster[i],i)
