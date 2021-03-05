import sys
import random
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings('ignore')

trainX, trainY = [], []
t_index = []
testX = []
trueY = []
newtestX = []
newtrainX = []


datafile = sys.argv[1]
file = open(datafile)
data = []
i = 0
l = file.readline()
while(l != '') : 
    a = l.split()
    t = len(a)
    l2 = []
    for j in range(0, t, 1):
        l2.append(float(a[j]))
        if j == (t-1) :
            l2.append(float(1))
    data.append(l2)
    l = file.readline()
rows = len(data)
cols = len(data[0])
file.close()


labelFile = sys.argv[2]
trainlabels = open(labelFile, encoding='utf-8').readlines()
trainlabels = [line.strip().split(' ') for line in trainlabels]
trainlabels = [list(map(int, line)) for line in trainlabels]


k = int(sys.argv[3])

     
def Z_i(data, M):   
    z = []
    for val in data:
        z.append(dp(val, M))
    return z

      
def W0(data, M):      
    dot = []
    for val in data:            
        dot.append(dp(val, M))
    min_dot = min(dot)
    max_dot = max(dot)
    return random.uniform(min_dot, max_dot)

        
def c_M(data):        
    M = []
    for _ in range(len(data[0])):      
        M.append(random.uniform(-1, 1))
    w0 =W0(data, M)
    NWs = [w0]
    NWs.extend(M)
    return NWs

def getbestC(train,labels):            
    random.seed()
    allCs = [.001, .01, .1, 1, 10, 100]
    error = {}
    for j in range(0, len(allCs), 1):
            error[allCs[j]] = 0
    rowIDs = []
    for i in range(0, len(train), 1):
            rowIDs.append(i)
    num_splits = 10
    for x in range(0,num_splits,1):        
            
            Ntrain = []
            Nlabels = []
            valid = []
            validlabels = []
            random.shuffle(rowIDs)      
            
            for i in range(0, int(.9*len(rowIDs)), 1):
                    Ntrain.append(train[i])
                    Nlabels.append(labels[i])
            for i in range(int(.9*len(rowIDs)), len(rowIDs), 1):
                    valid.append(train[i])
                    validlabels.append(labels[i])
            
            for j in range(0, len(allCs), 1):
                    C = allCs[j]
                    clf = LinearSVC(C=C)
                    clf.fit(Ntrain, Nlabels)
                    predict = clf.predict(valid)
                    err = 0
                    for i in range(0, len(predict), 1):
                            if(predict[i] != validlabels[i]):
                                    err = err + 1
                    err = err/len(validlabels)
                    error[C]+=err
                   
                    
    bestC = 0
    min_err=100
    keys = list(error.keys())
    for i in range(0, len(keys), 1):
           key = keys[i]
           error[key] = error[key]/num_splits
           if(error[key] < min_err):
                   min_err = error[key]
                   bestC = key
    return [bestC,min_err]
 
def dp(q1, q2):
    dp_1 = 0
    refw = q1
    refx = q2
    for j in range (cols):
        dp_1 += refw[j] * refx[j]
    return dp_1
   
def mini(val):  
    if val < 0 :
        return -1
    else:
        return 1
    
def column(zi):        
    colm = []
    for val in zi:
        num = 1 + mini(val)
        colm.append(num/2)
    return colm


for (label, i) in trainlabels:
    trainX.append(data[i])
    if label == 1:
        trainY.append(1)
    elif label == 0:
        trainY.append(-1)
    t_index.append(i)
TI = [i for i in range(len(data)) if i not in t_index]
if len(TI) == 0:
    TI = t_index
else:
    for i in TI:
        testX.append(data[i])
if len(testX) == 0:
    testX = trainX

for X in trainX:
    sic = [1.0]
    sic.extend(X)
    newtrainX.append(sic)
for X in testX:
    sic = [1.0]
    sic.extend(X)
    newtestX.append(sic)
Z_train, Z_test = [], []
for _ in range(k):
    M = c_M(testX)
    zi = Z_i(newtrainX, M)
    Ncolm = column(zi)
    Z_train.append(Ncolm)   
    zi = Z_i(newtestX, M)
    Ncolm = column(zi)
    Z_test.append(Ncolm)
NtrainX = list(map(list, list(zip(*Z_train))))
NtestX = list(map(list, list(zip(*Z_test))))
best_C, min_error = getbestC(NtrainX, trainY)
clf = LinearSVC(C= best_C, max_iter=10000)
clf.fit(NtrainX, trainY)
predict = clf.predict(NtestX)


for i, temp in enumerate(predict):
    if temp == -1:
        predict[i] = 0
for pred, index in zip(predict, TI):
    print(pred, index)



