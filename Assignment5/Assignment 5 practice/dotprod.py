
def transpose(m):
    for row in m : 
        print(row) 
    Trans  = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
    return Trans

def tran(m):
    for row in m:
        print(row)
    
    trans=[[0 for i in range(len(m))] for j in range (len(m[0]))] 
    print(trans)
    print('****')
    
    for i in range(len(m[0])):
        for j in range(len(m)):
            trans[i][j] = m[j][i]
    return trans
            
    
# Dot Product Function
def dot_Prod(m,n,c):
	dp=0
	for j in range(0, c, 1):
		dp += m[j]*n[j]
	return dp

a=[[1,4],[2,3],[6,8]]
b=[2,3]


d=transpose(a)
print(d)
 

c=dot_Prod(b,b,2) 
print(c)
 

e=tran(a)
print(e)