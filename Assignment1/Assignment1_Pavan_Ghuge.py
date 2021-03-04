import sys
# Data File Reading
testdatafile = sys.argv[1]
f1 = open(testdatafile)
data = []
i = 0
l = f1.readline()
while(l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    data.append(l2)
    l = f1.readline()
n_row = len(data)
n_col = len(data[0])
f1.close()

# Read training labels file and defining it as dictionary
trainlabelfile = sys.argv[2]
f2 = open(trainlabelfile)
trainlabels = {}
l = f2.readline()
while (l != ''):
    a = l.split()
    trainlabels[int(a[1])] = int(a[0])
    l = f2.readline()
f2.close()

# Calculating  Mean
def means(data1,traindata,label):
    count_labels=0
    mean=[0.1]*len(data1[0])
    for i in range (0, len(data1), 1):
        if(traindata.get(i) != None and traindata[i] == label):
            for j in range(0,len(data1[0]),1):
                mean[j] = mean[j]+ float(data1[i][j]) 
            count_labels +=1
 
    for j in range(0,len(data1[0]),1):    
        mean[j]= mean[j]/count_labels
    return mean
   
# Calculating  Variance
def variance(data1,traindata,label,mean):
    var=[0.0]*len(data1[0])
    for i in range (0, len(data1), 1):
        if(traindata.get(i) != None and traindata[i] == label):
            for j in range(0,len(data1[0]),1):
                var[j] = var[j]+ (data1[i][j]-mean[j])**2 
    return var

# Calculating  Standard Deviation
def std_dev(Variance,data1,traindata,label):
      sd=[0.0]*len(data1[0])
      count_labels = 0
      
      for i in range (0, len(data1), 1):
        if(traindata.get(i) != None and traindata[i] == label):
            count_labels +=1
      
      for j in range(0,len(data1[0]),1):
          sd[j] = (Variance[j]/count_labels)**0.5
      return sd   

#Calling Means  
m0=means(data,trainlabels,0)
m1=means(data,trainlabels,1)

#Calling Variances
v0=variance(data,trainlabels,0,m0)
v1=variance(data,trainlabels,1,m1)

#Calling Standard Deviations
sd0=std_dev(v0,data,trainlabels,0)
sd1=std_dev(v1,data,trainlabels,1)
 
print("Predicted lables for row ID")  
for i in range(0, n_row, 1):
    if (trainlabels.get(i) == None):
        dev_0 = 0
        dev_1 = 0
        for j in range (0, n_col, 1):
            dev_0 = dev_0 + ((data[i][j] - m0[j])/sd0[j])**2
            dev_1 = dev_1 + ((data[i][j] - m1[j])/sd1[j])**2 
        if (dev_0 < dev_1):
            print("0",i)
        else:
            print("1",i)




    

    
    
    
    
    
    
    
    
    
