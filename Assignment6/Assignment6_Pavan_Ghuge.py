import sys

dataFile = sys.argv[1]
labelFile = sys.argv[2]


#data = open("/Users/pavanghuge/Downloads/ML/Test Data/ionosphere/ionosphere.data")
data =open(dataFile)
data = [line.strip().split() for line in data]
data = [list(map(float, line)) for line in data]

#trainlabels = open("/Users/pavanghuge/Downloads/ML/Test Data/ionosphere/ionosphere.trainlabels.0")
trainlabels = open(labelFile)
trainlabels = [line.strip().split(' ') for line in trainlabels]
trainlabels = [list(map(int, line)) for line in trainlabels]

features, labels = [], []
labels_Index = []

for (label, i) in trainlabels:
    features.append(data[i])


    if label == 1:
        labels.append(1)
    else:
        labels.append(-1)

    labels_Index.append(i)
# Check if data rows are equal to label rows
testX = []
testIndex = [i for i in range(len(data)) if i not in labels_Index]

if len(testIndex) == 0:
    testIndex = labels_Index

else:
    for i in testIndex:
        testX.append(data[i])

if len(testX) == 0:
    testX = features

###### transpose function
def transpose(m):
        Trans = [[m[j][i] for  j in range(len(m))] for i in range(len(m[0]))]
        return Trans
   

def calc_gini(features,labels):
    
    gini={}
    rows =len(features)
    colm_features = transpose(features)
    
    for i , colm in enumerate(colm_features):
        for split_index in range(0,rows,1):
            
            left_labels, right_labels =[], []
            split_val = colm[split_index]
            
            for j, val in enumerate(colm):
                if val < split_val:
                    left_labels.append(labels[j])
                else:
                    right_labels.append(labels[j])
                    
            lsize =len(left_labels)
            rsize =len(right_labels)
         
            
            lp = 0 
            rp = 0
           
######## count labels in left node ######       
            for lab in left_labels:
                if lab ==-1:
                    lp +=1
                else:
                    lp +=0
                    
####### count labels in right node #####
            for lab in right_labels:
                if lab ==-1:
                    rp +=1
                else:
                    rp +=0
            
####### gini value #####            
            if lsize ==0:
                term1 =0
                
            else:
                term1 = (lp/rows)*(1-(lp/lsize))
                    
            if rsize ==0:
                term2 =0
                
            else:
                term2 = (rp/rows)*(1-(rp/rsize))
                
            gini_coef = term1 + term2


####### Selecting feature no ###########             
            left_features = []
            for val in colm:
                if val<split_val:
                    left_features.append(val)
                    
            if len(left_features)!=0:
                left_max=max(left_features)
            else:
                left_max =split_val
                    
            main_split=(left_max + split_val)/2
                
            if gini_coef not in gini.keys():
                gini[gini_coef]=[i,main_split]
            else:
                gini[gini_coef].append([i,main_split])
                
    
    min_gini = min(gini.keys())         

    return gini[min_gini]


sp = calc_gini(features,labels)
print("Column number =", sp[0],"Gini Value =", sp[1])






















