import sys
import random

data=open("/Users/pavanghuge/Downloads/ML/Test Data/ionosphere/ionosphere.data")
features=[]
i=0
l=data.readline()

#reading data from data file and adding 1 to each row
while(l!=''):
    split_data = l.split()
    split_data_len =len(split_data)
    l2=[]

######adding 1 to each row for getting intercept
    for i in range(0,split_data_len,1):
        l2.append(float(split_data[i]))
        if i ==(split_data_len-1):
            l2.append(float(1))
######adding split data and 1 to features    
    features.append(l2)
    l = data.readline()

rows =len(features)
cols=len(features[0])

#reading labels from label file and converting 0 to -1        

trainlabels=open("/Users/pavanghuge/Downloads/ML/Test Data/ionosphere/ionosphere.trainlabels.0")
labels={}
count_labels=[0]*2
l = trainlabels.readline()
 
while(l!=''):
    split_labels= l.split()
    if int(split_labels[0])==0:
        labels[int(split_labels[1])] = -1
    else:
        labels[int(split_labels[1])] = int(split_labels[0])
    count_labels[int(split_labels[0])] +=1    
    l = trainlabels.readline()
    
###### transpose function
def transpose(m):
        Trans = [[m[j][i] for  j in range(len(m))] for i in range(len(m[0]))]
        return Trans

###### dot product function
def dot_Prod(m,n,c):
    dp=0
    for j in range(0,c,1):
        dp += m[j]*n[j]
    return dp

###### change in weights 
def delf(features,labels,weights):
    
    rows =len(features)
    cols=len(features[0])
     #### update w
    delta_weights=[0]*cols
    for i in range(0, rows, 1):
            if (labels.get(i) != None):
                dp = dot_Prod(weights, features[i], cols)
                for j in range(0, cols, 1):
                    if(dp*labels.get(i)<1):
                        delta_weights[j] += (-labels.get(i)*features[i][j])
                    else:
                        delta_weights[j]+=0
    
    return delta_weights

###### error function
def hinge_loss(features,labels,weights):
    
    rows =len(features)
    cols=len(features[0])
    error = 0
    for i in range(0, rows, 1):
        if (labels.get(i) != None):
            error += max(0,(1-labels.get(i)*dot_Prod(weights, features[i], cols)))
    
    return error 

   
######### best learning rate 
def bestEta(features,labels,weights,eta_list):
    
    eta_error ={}
    error = 0
    new_weights  = [0]*(len(weights))
    
    for eta in eta_list:
        del_f = delf(features,labels,weights)
    
        for j in range(0, cols, 1):
            new_weights[j] = weights[j] - eta * del_f[j]
            
        error= hinge_loss(features,labels,new_weights)
        
        eta_error[error]= eta
    
    return eta_error[min(eta_error.keys())]


######### MODEL TRAINING


cond =1
error = 0
theta =0.001
counter = 0
weights = [0]* cols
eta_list = [1, .1, .01, .001, .0001, .00001, .000001,
            .0000001, .00000001, .000000001, .0000000001,
            .00000000001 ]
#### Initailizing weights to random    
for j in range(cols):
    weights[j] = random.uniform(-0.001,0.001)

########## error initialization   
  
error = hinge_loss(features,labels,weights)

while(True):
    counter += 1
    
    del_f = delf(features,labels,weights)
    eta =  bestEta(features,labels,weights,eta_list)
    for i,df in enumerate(del_f):
        weights[i] -= eta*df
    
    new_error = hinge_loss(features,labels,weights)
    
    if error - new_error < theta:
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
   
    
    
    
    
    



        








