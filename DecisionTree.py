# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 23:28:28 2018

@author: yisu.tian
"""


import pandas as pd
from collections import Counter
from math import log
from sklearn import metrics
import random
from copy import deepcopy
import operator  


### data input

train1=pd.read_csv("D:/training_set1.csv")
val1=pd.read_csv("D:/validation_set1.csv")  
test1=pd.read_csv("D:/test_set1.csv") 

train2=pd.read_csv("D:/training_set2.csv")
val2=pd.read_csv("D:/validation_set2.csv")  
test2=pd.read_csv("D:/test_set2.csv") 



# define a Decision Tree. 
# att: attribute
# results: the dictionary contains data counter in each leaf node
class DecisionTree:
   
    def __init__(self, att=-1, Branch_left=None, Branch_right=None, results=None):
        self.att=att
        self.Branch_left=Branch_left
        self.Branch_right=Branch_right
        self.results=results
       

# Entropy function which used to select the attribute in each step when building the tree
def Entropy(data):
    
    results=Counter(data['Class'])
    E=0
    for key in results:
        pro=results[key]/len(data)
        Log=log(pro)/log(2)
        E+=pro*Log
    return -E


# Variance imputiry
def VI(data):
  
    results=Counter(data['Class'])
    if len(results)<2:
        return 0
    else:
        V=1
        for key in results:
            V*=results[key]/len(data)
        return V
    
    
#separate data set based on Class=0 or 1
def dividedata(data, column):

    list0=data.loc[data[column]==0]
    list1=data.loc[data[column]==1]
    return (list0, list1)




# Build the Decision Tree:
# Input: data, the specific method used for selecting the attribute in each step
# Output: Decision Tree 

def Building(data, Heu):
    
    def BuildDT(data, Heu,a=0):
        if a>((len(list(data))-1)/2)+1:
            ress=dict(Counter(data['Class']))
            r=sorted(ress.items(), key=operator.itemgetter(0),reverse=True)[0]
            return DecisionTree(results={r[0]:10})
        elif len(data)==0:
            return DecisionTree()
        else:
            BestGain=0
            RHeu=Heu(data)
            for att in list(data)[:-1]:
                list0, list1=dividedata(data, att)
                prob0=len(list0)/len(data)
                Gain=RHeu-prob0*Heu(list0)-(1-prob0)*Heu(list1)
                    #Gain function used to select the attrubute in each step
                if BestGain<Gain and len(list0)*len(list1)!=0:
                    BestGain=Gain
                    BestLists=(list0, list1)
                    BestAtt=att
            if BestGain>0: 
                a+=1        # if BestGain>0 means we need to put an Non-leaf node
                Branch_left=BuildDT(BestLists[0],Heu,a)
                Branch_right=BuildDT(BestLists[1],Heu,a)
                return DecisionTree(att=BestAtt, results=dict(Counter(data['Class'])),Branch_left=Branch_left, Branch_right=Branch_right)
            # note the attribute name, results and the specific data sets in this node
            else: 
            # if this is a leaf, we just need to add results in the decision tree
                return DecisionTree(results=dict(Counter(data['Class'])))
    return BuildDT(data, Heu,a=0)
# select all nonleaf nodes into a list
# it will be used in pruning         
def NonLeafs(tree):
    a=[]
    def q(tree):
        if len(tree.results)!=1:
            a.append(tree)
            q(tree.Branch_left)
            q(tree.Branch_right)
    q(tree)        
    return a
   

            
# Pruning the Decision Tree to avoid overfitting and improve the accurancy
# Input: decision tree; validation data for pruning, 
#        integer L: number of pruning trees 
#        integer K: pruning times in each pruning tree 
def Prune(tree, val, L, K):   
    
    Tbest=deepcopy(tree) # initial tree
    res_old=metrics.accuracy_score(val['Class'], Classify(val, Tbest))
    for i in range(L):
        tree1=deepcopy(Tbest)
        M=random.randint(1,K)
        for j in range(M):
            t1=NonLeafs(tree1) # recreate Non-leafs list after one pruning
            if len(t1)!=0: 
                P=random.randint(0, len(t1)-1) #randomly pick one leaf with index
                ele=sorted(t1[P].results.items(), key=operator.itemgetter(0),reverse=True)[0]
                t1[P].results={ele[0]:ele[1]} 
                # assign the majority class of the subset of the data to the leaf node
                t1[P].Branch_left,t1[P].Branch_right=None, None
            else: # if the tree not existed after pruning, just break
                break
        res_test=metrics.accuracy_score(val['Class'], Classify(val, tree1))
        if res_test>res_old:
            Tbest=deepcopy(tree1)
            res_old=res_test
    return Tbest                  
                
# print the Decision Tree                
def printtree(tree,indent=''):
  
    if len(tree.results)==1: # if results in one node just contains one key variable, it is a leaf node
        print (list(tree.results)[0])
    else:
        print()
        
        print ( indent+str(tree.att)+' = 0 '+' : ',end='')
        printtree(tree.Branch_left,indent+'|')
        print ( indent+str(tree.att)+' = 1 '+' : ',end='')
        printtree(tree.Branch_right,indent+'|')         

    
    

def Classify(aim, tree):
    
    def C(v, tree):
        if len(tree.results)==1:
            
            return list(tree.results)[0]
        else:
            vi=int(v[tree.att])
            if vi==0:
                b=tree.Branch_left
            else:
                b=tree.Branch_right
        return C(v, b)
    pred=[]
    for i in range(0,len(aim)):
        v=aim.iloc[i:i+1,:-1]
        pred.append(C(v,tree))
    return pred
            

    
    
  
    
    




def Testing(K, L, train, val, test, to_print='Yes'):
    T11=Building(train, VI)
    T12=Prune(T11,val, L, K)
    T21=Building(train, Entropy)
    T22=Prune(T21,val, L, K)
    accurancy11=metrics.accuracy_score(test['Class'], Classify(test, T11))
    accurancy12=metrics.accuracy_score(test['Class'], Classify(test, T12))
    accurancy21=metrics.accuracy_score(test['Class'], Classify(test, T21))
    accurancy22=metrics.accuracy_score(test['Class'], Classify(test, T22))
    print ("The accurancy of this Decision Tree using Variance impurity is: "+ str(accurancy11))
    print ("After pruning, the accurancy of this Decision Tree using Variance impurity is： "+ str(accurancy12))
    print ("The accurancy of this Decision Tree using Entropy is "+ str(accurancy21))
    print ("After pruning, the accurancy of this Decision Tree using Entropy is: "+ str(accurancy22))
    if to_print=='Yes':
        print ('The Decision Tree using Variance impurity:')
        printtree(T11)
        print ('The pruned Decision Tree using Variance impurity:')
        printtree(T12)
        print ('The Decision Tree using Entropy:')
        printtree(T21)
        print ('The pruned Decision Tree using Entropy:')
        printtree(T22)



def Show_pruning(train, val, test):
    T1=Building(train, VI)
    T2=Building(train, Entropy)
    a1=metrics.accuracy_score(test['Class'], Classify(test, T1))
    a2=metrics.accuracy_score(test['Class'], Classify(test, T2))
    print ('The accurancy using Variance impurity before pruning is '+str(a1))
    print ('The accurancy using Entropy before pruning is '+str(a2))
    set1=[]
    set2=[]
    k=[]
    l=[]
    for i in range(10):
        K=random.randint(1,20)
        L=random.randint(1,20)
        k.append(K)
        l.append(L)
        t1=Prune(T1,val, L, K)
        t2=Prune(T2,val, L, K)
        accurancy1=metrics.accuracy_score(test['Class'], Classify(test, t1))
        accurancy2=metrics.accuracy_score(test['Class'], Classify(test, t2))
        set1.append(accurancy1)
        set2.append(accurancy2)
    res = { 'K':k,'L':l, 'Entropy': set2,'Variance Impurity': set1}
    result = pd.DataFrame(data=res)
    
    return result



