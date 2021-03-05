import sys
import math
import random
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score




################ DATA PRE-PROCESSING ################
dataFile = sys.argv[1]
labelFile = sys.argv[2]
trainFile = sys.argv[3]

#data = open("/Users/pavanghuge/Downloads/ML/Assignment9A/qsar.data").readlines()
data =open(dataFile)
data = [line.strip().split() for line in data]
data = [list(map(float, line)) for line in data]
rows = len(data)
cols = len(data[0])

#labels = open("/Users/pavanghuge/Downloads/ML/Assignment9A/qsar.labels").readlines()
labels = open(labelFile)
labels = [line.strip().split(' ') for line in labels]
labels = [list(map(int, line)) for line in labels]


#trainlabels = open("/Users/pavanghuge/Downloads/ML/Assignment9A/qsar.trainlabels.0").readlines()
trainlabels = open(trainFile)
trainlabels = [line.strip().split(' ') for line in trainlabels]
trainlabels = [list(map(int, line)) for line in trainlabels]




trainX1, trainY1 = [], []
trainIndex = []

for (label, i) in trainlabels:
    trainX1.append(data[i])

    if label == 1:
        trainY1.append(1)
    else:
        trainY1.append(0)

    trainIndex.append(i)

testX1 = []
testIndex = [i for i in range(len(data)) if i not in trainIndex]

if len(testIndex) == 0:
    testIndex = trainIndex

else:
    for i in testIndex:
        testX1.append(data[i])

if len(testX1) == 0:
    testX1 = trainX1



testY1=[]
testIndex = [i for i in range(len(labels)) if i not in trainIndex]

if len(testIndex) == 0:
    testIndex = trainIndex

else:
    for val,i in labels:
        if i in testIndex:
            testY1.append(val)
            




def create_train_test(trainX, testX,k):
        
    if len(trainX[0]) != len(testX[0]):
        raise ValueError('Training and Testing data should contain same number of features')
        
    new_trainX = []

    for X in trainX:
        temp = [1.0]
        temp.extend(X)
        new_trainX.append(temp)
        
    new_testX = []
    for X in testX:
        temp = [1.0]
        temp.extend(X)
        new_testX.append(temp)
                
    Z_train, Z_test = [], []
    for i  in range(k):              
        weights = create_weights(trainX)
            
        zi = create_zi(new_trainX, weights)
        new_colm = get_new_column(zi)
        Z_train.append(new_colm)
            
        zi = create_zi(new_testX, weights)
        new_colm = get_new_column(zi)
        Z_test.append(new_colm)
        
    Z_train = list(map(list, list(zip(*Z_train))))
    Z_test = list(map(list, list(zip(*Z_test))))
        
    return Z_train, Z_test


def dot_product(P, Q):
    res = 0
    for val_a, val_b in zip(P, Q):
        res += val_a * val_b
        
    return res
    
def get_w0(data, weights):
    dot = []
        
    for val in data:
            
        dot.append(dot_product(val, weights))
        
    min_dot = min(dot)
    max_dot = max(dot)
        
    return random.uniform(min_dot, max_dot)
            
def create_weights(data):
        
    weights = []
        
    for _ in range(len(data[0])):
            
        weights.append(random.uniform(-1, 1))
        
    w0 = get_w0(data, weights)
        
    new_weights = [w0]
        
    new_weights.extend(weights)
        
    return new_weights

def create_zi(data, weights):
        
    zi = []
        
    for val in data:
        zi.append(dot_product(val, weights))
        
    return zi
    
def sign(val):
        
    if val < 0 :
        return -1
        
    else:
        return 1

def get_obj(val):
        
    num = 1 + sign(val)
        
    return num/2
    
def get_new_column(zi):
        
    colm = []
        
    for val in zi:
        colm.append(sign(val))
            
    return colm

def getbestC(train,labels):
                
    random.seed()
    allCs = [.001, .01, .1, 1, 10, 100]
    error = {}
    for j in range(0, len(allCs), 1):
        error[allCs[j]] = 0
    rowIDs = []
    for i in range(0, len(train), 1):
        rowIDs.append(i)
    nsplits = 10
    for x in range(0,nsplits,1):        
#### Making a random train/validation split of ratio 90:10
        newtrain = []
        newlabels = []
        validation = []
        validationlabels = []

        random.shuffle(rowIDs) #randomly reorder the row numbers      
                #print(rowIDs)

        for i in range(0, int(.9*len(rowIDs)), 1):
            newtrain.append(train[i])
            newlabels.append(labels[i])
        for i in range(int(.9*len(rowIDs)), len(rowIDs), 1):
            validation.append(train[i])
            validationlabels.append(labels[i])

#### Predict with SVM linear kernel for values of C={.001, .01, .1, 1, 10, 100} ###
        for j in range(0, len(allCs), 1):
            C = allCs[j]
            clf = LinearSVC(C=C)
            clf.fit(newtrain, newlabels)
            prediction = clf.predict(validation)
                        
            err = 0
            for i in range(0, len(prediction), 1):
                if(prediction[i] != validationlabels[i]):
                    err = err + 1

            err = err/len(validationlabels)
            error[C]+=err
          
    bestC = 0
    minerror=100
    keys = list(error.keys())
    for i in range(0, len(keys), 1):
        key = keys[i]
        error[key] = error[key]/nsplits
        if(error[key] < minerror):
            minerror = error[key]
            bestC = key

    return [bestC,minerror]






newdata_train =[]

#filename= sys.argv[3]
#f = open(filename,'w')
f = open("results_qsar.txt","w")
best_C, min_error = getbestC(trainX1, trainY1)  
clf = LinearSVC(C= best_C, max_iter=10000)
clf.fit(trainX1, trainY1)
org_predictions = clf.predict(testX1)
org_error = 1-balanced_accuracy_score(testY1, org_predictions)

planes=[10, 100, 1000,10000]

for k in planes:
    new_trainX, new_testX = create_train_test(trainX1, testX1,k)
    best_C, min_error = getbestC(new_trainX, trainY1)        
    clf = LinearSVC(C= best_C, max_iter=10000)
    clf.fit(new_trainX, trainY1)
    predictions = clf.predict(new_testX)
    balanced_error=1-balanced_accuracy_score(testY1, predictions)
    
    print("\n******************************************************\n")
    print("For",k,"random planes")
    print("Best C:",best_C)
    print("Error for new features data: \n",balanced_error*100,"%")
    print("Error for original data: \n",org_error*100,"%")
    print("\n******************************************************\n")
    
    f.write("\n******************************************************* \n")
    f.write("For %d random planes \n" %k)
    f.write(" Best C:"+ "\n")
    f.write(str(best_C)+"\n")
    f.write("\n")
    f.write("Error for new features data "+ "\n")
    f.write(str(balanced_error*100) +"\n")
    f.write("\n")
    f.write(" Error for original data: "+ "\n")
    f.write(str(org_error*100) +"\n")
    f.write("\n*******************************************************\n")
   
f.close()
   



