import random

class getTheGinie:
    def __init__(self):
        pass
    
    def grantMyWish(self, trainX, trainY):
        
        '''
        
        input : (list, list)
        output : (int, int)   
        
        This function returns the column and threshold with minimum gini value 
        for a given data.
        
        '''
        
        lamp = {}
        
        rows = len(trainX)
        colm_trainX = list(zip(*trainX))
        
        for i, colm in enumerate(colm_trainX):
            for split_index in range(0, rows, 1):
                
                left_trainY, right_trainY = [], []
                
                split_val = colm[split_index]
                
                for j, val in enumerate(colm):
                    if val < split_val:
                        left_trainY.append(trainY[j])
                    else:
                        right_trainY.append(trainY[j])
                
                lsize = len(left_trainY)
                rsize = len(right_trainY)
                
                lp = sum([1 if lab == -1 else 0 for lab in left_trainY])
                rp = sum([1 if lab == -1 else 0 for lab in right_trainY])

                if lsize == 0:
                    term1 = 0
                else:
                    lterm_1 = lsize/rows
                    lterm_2 = lp/lsize
                    lterm_3 = 1 - (lp/lsize)
                    
                    term1 = lterm_1 * lterm_2 * lterm_3
                
                if rsize == 0:
                    term2 = 0
                else:
                    rterm_1 = rsize/rows
                    rterm_2 = rp/rsize
                    rterm_3 = 1 - (rp/rsize)
                    
                    term2 = rterm_1 * rterm_2 * rterm_3
                
                ginie =  term1 + term2
                
                left_trainX = []
                for val in colm:
                    if val < split_val:
                        left_trainX.append(val)
                
                if len(left_trainX) != 0:
                    left_max = max(left_trainX)
                else:
                    left_max = split_val
                
                main_split = (left_max + split_val)/2
                
                lamp[ginie] = [i, main_split]
                
        wish = min(lamp.keys())
        
        return lamp[wish]
    
    def getBootstrappedData(self, data, labels):
        
        '''
        
        input : (list, list)
        output : (list, list)
        
        This function returns the bootstrapped data for a given data.
        
        '''
        
        bootstrapped_data = []
        bootstrapped_labels = []
        
        for _ in range(int(len(data))):
            idx = random.randint(0, len(data)-1)
            bootstrapped_data.append(data[idx])
            bootstrapped_labels.append(labels[idx])
        
        return bootstrapped_data, bootstrapped_labels
    
    def getLabels(self, trainX, trainY, split_colm, split_val):
        
        '''
        
        input : (list, list, int, float)
        output : (list)
        
        This function returns the labels for the left and the right partition
        based on the majority from training data.
        
        '''
        
        left, right = 0, 0
        
        train_colm = list(list(zip(*trainX))[split_colm])
        
        for i, val in enumerate(train_colm):
            if val < split_val:
                left += trainY[i]
            else:
                right += trainY[i]
        '''
        if sum(left) >= 0:
            return [1, -1]
        else:
            return [-1, 1]
        '''
        
        if left >= 0:
            left_lab = 1
        else:
            left_lab = -1
        
        if right >= 0:
            right_lab = 1
        
        else:
            right_lab = -1
        
        return [left_lab, right_lab]
    def bagging(self, trainX, trainY, testX, iterations = 100):
        
        '''
        
        input : (list, list, list, int)
        output : (list)
        
        This function implements bagging and returns prediction on test data.
        
        '''
       
        if len(trainX) != len(trainY):
            raise ValueError('Data and labels dimensions mismatch')
        
        if type(iterations) != int:
            raise ValueError('iterations should be interger type')
        
        predictions = [0] * len(testX)
        
        for _ in range(iterations):
            
            data, labels = self.getBootstrappedData(trainX, trainY)
            
            split_colm, split_val = self.grantMyWish(data, labels)
            
            left_label, right_label = self.getLabels(data, labels, split_colm, split_val)
            
            testX_colm = list(list(zip(*testX))[split_colm])
            
            for i, val in enumerate(testX_colm):
                if val < split_val:
                    predictions[i] += left_label
                else:
                    predictions[i] += right_label
        
        results = []
        
        for i, pred in enumerate(predictions):
            if pred >= 0:
                results.append(1)
            else:
                results.append(0)
        
        return results
            