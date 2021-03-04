
import sys
import random
import math


eta=0.001
theta=0.001


#reading data from file
features= sys.argv[1]
f = open(features)
data1 = []
i = 0
list = f.readline()

while(list != ''): 
    a1 = list.split() 
    alength=len(a1)
    list2 = [] #new list is been created in which we can append the data
    for j in range(0, alength, 1):
        list2.append(float(a1[j]))
        if j ==(alength-1):
            list2.append(float(1))
    data1.append(list2)
    list = f.readline()
    
#length of rows and columns are calculated
rows = len(data1)
cols = len(data1[0])
f.close()

outputfile = sys.argv[2]
f = open(outputfile)
trainlabels = {}
n1 = []
n1.append(0)
n1.append(0)
l = f.readline()

#swapping the key position of trainlabels
while (l != ''):
    a1 = l.split()
    if int(a1[0])==0:
        trainlabels[int(a1[1])] = -1
    else:
        trainlabels[int(a1[1])] = int(a1[0])
    l = f.readline()
    n1[int(a1[0])] +=1

#function for dot product
def dot_prod(m,n ,c):
    dotp=0
    
    for j in range(0, c, 1):
        dotp += m[j]*n[j]
    return dotp


#intitilizing random weights, counter, terminating conditions , learning rate and error
    
weight = [0]*cols

condition=0
counter=0

error=0


for j in range(0, cols, 1):
    weight[j] = (0.02 * random.uniform(0,1)) - 0.01
    



#calculating hinge error
for i in range(0, rows, 1):
	if (trainlabels.get(i) != None):
		error =error+ max(0,(1-trainlabels.get(i)*dot_prod(weight, data1[i], cols)))
        

while(condition!=1):
    counter=counter+1
    delf=[0]*cols
    for i in range(0,rows,1):
        if trainlabels.get(i)!=None:
            dp=dot_prod(weight,data1[i],cols)
            for j in range(0,cols,1):
                if(dp*trainlabels.get(i)<1):
                    delf[j]+=-1*data1[i][j]*trainlabels.get(i)
                else:
                    delf[j]+=0
            
        
#updating the weights
    for j in range(0, cols, 1):
        weight[j] = weight[j] - eta*delf[j]
        
#compute hinge error new
    new_error = 0;
    for i in range(0,rows,1):
        if (trainlabels.get(i) != None):
            new_error +=max(0,(1-trainlabels.get(i)*dot_prod(weight, data1[i], cols)))
    

    if abs(error - new_error) < theta:
        condition = 1
    error = new_error
            
print("Predicted Output")
#predictions
for i in range(0, rows, 1):
	if (trainlabels.get(i) == None):
		dotprod = dot_prod(weight, data1[i], cols)
		if(dotprod>0):
			print("1",i)
		else:
			print("0",i)  

# distance from origin
sum2=0
for j in range(0,cols-1,1):
    sum2= float(sum2)+weight[j]**2               
#calculating the square root 
sum2=math.sqrt(sum2)
#distance from origin
distance=0
distance=abs(weight[len(weight)-1]/sum2)
print("distance from origin:",distance)
            