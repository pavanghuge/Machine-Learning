import sys
import random
import math


#Read Data
#data = sys.argv[1]
f=open("/Users/pavanghuge/Downloads/ML/Assignment4/ionosphere.data")

features = []
l = f.readline()
while (l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    l2.append(float(1))
    features.append(l2)
    l = f.readline()

rows = len(features)
cols = len(features[0])
f.close()
#Read Trainlabels
#trainlabels = sys.argv[2]
#f = open(trainlabels)
f=open("/Users/pavanghuge/Downloads/ML/Assignment4/ionosphere.trainlabels.0")
labels = {}
n = [0, 0]
l = f.readline()
while (l != ''):
    a = l.split()
    labels[int(a[1])] = int(a[0])    
    l = f.readline()
    n[int(a[0])] += 1
f.close()


#Initializing weights
weights = [0]*cols
for j in range(0, cols, 1):
    # print(random.random())
    weights[j]= random.uniform(-0.001,0.001)

# Transpose of matrix function
def transpose(m):
    for row in m : 
        print(row) 
    Trans  = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
    return Trans
#For transpose of 1D vector
def List_Transpose(m):
    Trans  = [m[i] for i in range(len(m))]
    return Trans

#Dot product
def dot_Prod(a, b):
    dp = 0
    for i in range(0, cols, 1):
        dp += a[i] * b[i]
    # dp = sum(p*q for p,q in zip(a, b))
    return dp

#Initial Values of important Metrics
eta = 0.001
error = 0
error_diff = 1
count = 0
theta = 0.001     
counter =0


while (error_diff > theta):
    counter +=1
    delta_weights = [0] * cols
    for j in range(0, rows, 1):
        if (labels.get(j) != None):
            dp = dot_Prod(weights, features[j])
            expo = (labels.get(j)) - (1 / (1 + (math.exp(-1 * dp))))
            for k in range(0, cols, 1):
                delta_weights[k] += (expo) * features[j][k]
              
#Calculating Weights 
    for j in range(0, cols, 1):
        weights[j] += eta * delta_weights[j]
    prev = error
    error = 0

#gradient descent
    for j in range(0, rows, 1):
        if (labels.get(j) != None):
            error += math.log(1 + math.exp((-1 * (labels.get(j))) * (dot_Prod(weights, features[j]))))
           
    error_diff = abs(prev - error)
        

'''print ("Error = ",error)
    
print("Final Weights = ",weights)
print("\n\n") 
print("Loop count = ",counter)
print("Final Error=",error)
print("Predicted   Row ID")
print("Output")'''

#Prediction 
for i in range(0, rows, 1):
    if (labels.get(i) == None):
        dp = dot_Prod(weights, features[i])
        if (dp > 0):
            print("1", i)
        else:
            print("0", i)

#Distance from Origin
normq = 0
for i in range(0, (cols - 1), 1):
    normq += weights[i] ** 2
normq = math.sqrt(normq)
d = (weights[len(weights) - 1] / normq)
print("Distance to origin = ",d)
print("Eta and Theta are set for Ionosphere Data")

