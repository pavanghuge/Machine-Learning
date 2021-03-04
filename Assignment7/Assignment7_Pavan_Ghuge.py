import sys
import random

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


def bagging(features,labels,test_features):
    
    
    predictions= [0]*len(features)
    iterations =100
    for _ in range(iterations):
        
        B_fearures , B_labels = bootstrap(features,labels)
        
        split_col, split_val = calc_gini(features, labels)
        
        left_label, right_label = label_count(features,labels,split_col, split_val)
        
        test_feature_col =list(list(zip(*test_features))[split_col])
        
        
        for i, val in enumerate(test_feature_col):
            if val < split_val:
                predictions[i] += left_label
            else:
                predictions[i] += right_label
                
        predicted_val = []
        
    for i, prediction in enumerate(predictions):
        if prediction >=0:
            predicted_val.append(1)
        else:
            predicted_val.append(0)
    
    return predicted_val

def bootstrap(features,labels):
    
    B_features = []
    B_labels = [] 
    
    for _ in range(int(len(features))):
        
        selected_data= random.randint(0,len(features)-1)
        
        B_features.append(features[selected_data])
        B_labels.append(labels[selected_data])
    
    return B_features,B_labels

def label_count(features,labels,split_col, split_val):
    
    left_count, right_count = 0, 0
    
    feature_col = list(list(zip(*features))[split_col])
        
    for i, val in enumerate(feature_col):
        if val < split_val:
            left_count += labels[i]
        else:
            right_count += labels[i]
        
        
    if left_count >= 0:
        left_lab = 1
    else:
        left_lab = -1
        
    if right_count >= 0:
        right_lab = 1
        
    else:
        right_lab = -1
        
    return [left_lab, right_lab]


predict_labels = bagging(features,labels,testX) 

for pred, index in zip(predict_labels,testIndex):
    print(pred, index)






















