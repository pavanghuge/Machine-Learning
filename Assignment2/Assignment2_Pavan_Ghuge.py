import sys
import random

data = sys.argv[1]
f1 = open(data)
features = []
i = 0
l = f1.readline()
# Reading data from datafile  and add 1 to each row
while(l != ''):
     split_data = l.split()
     split_data_len = len(split_data)
     l2=[]
#adding 1 to each 1 for getting the intercept 
     for i in range(0, split_data_len, 1):
         l2.append(float(split_data[i]))
         if i == (split_data_len-1):
             l2.append(float(1))
     features.append(l2)
     l = f1.readline()
rows = len(features)
cols = len(features[0])
f1.close()

# Reading Labels from file and converting 0 to -1
labelfile = sys.argv[2]
f2 = open(labelfile)
labels = {}
count_labels= [0]*2
l = f2.readline()
while(l != ''):
	split_data = l.split()
	if int(split_data [0]) == 0:
		labels [int(split_data[1])] = -1
	else:
		labels [int(split_data[1])] = int(split_data[0])
	l = f2.readline()
	count_labels[int(split_data[0])] += 1
f2.close()

# Transpose of matrix function
def transpose(m):
    for row in m : 
        print(row) 
    Trans  = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
    return Trans

# Dot Product Function
def dot_Prod(m,n,c):
	dp=0
	for j in range(0, c, 1):
		dp += m[j]*n[j]
	return dp

# Initialize weights to random 

weights = [0]*cols
for j in range(0, cols, 1):
	weights[j] = (random.uniform(-0.001,0.001))


eta = 0.0001
error = 0

# Intial error
for i in range(0, rows, 1):
	if (labels.get(i) != None):
		error += (-labels.get(i) + dot_Prod(weights,features[i], cols))**2

counter=0

while True:
    counter += 1
    print("error",error)
    delta_weights=[0]*cols
# Calculate Delta weights 
    for i in range(0, rows, 1):
        if (labels.get(i) != None):
            dp = dot_Prod(weights, features[i], cols)
            for j in range(0, cols, 1):
                delta_weights[j] += ((-labels.get(i) + dp)*features[i][j])


# New Weights
    for j in range(0, cols, 1):
        weights[j] = weights[j] - eta*delta_weights[j]

# New Error
    new_error = 0
    for i in range(0, rows, 1):
        if (labels.get(i) != None):
            new_error += (-labels.get(i) + dot_Prod(weights, features[i], cols))**2
    
    if (error - new_error) < 0.001:
        break
    error = new_error

print("Loop count = ",counter)
print("Final Weights = ",weights)
print("Final Error=",error)
print("Predicted Row ID")
print("Output")
# Prediction
for i in range(0, rows, 1):
	if (labels.get(i) == None):
		dp = dot_Prod(weights, features[i], cols)
		if(dp>0):
			print("   1      ",i)
		else:
			print("   0      ",i)
#distance from origin

def List_Transpose(m):
    Trans  = [m[i] for i in range(len(m))]
    return Trans

#w0 for calculation of distance from origin as it is weight for intercept
w0=weights[cols-1] 
w=[0]*(cols-1)
for j in range(0,cols-1,1):
    w[j]= weights[j]

w_t=List_Transpose(w)
w_p =dot_Prod(w_t,w,cols-1) 
dist_org=0
dist_org =w0/(w_p**0.5)
print("Distance from Origin=",dist_org)





















    
