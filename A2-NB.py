# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 13:07:16 2018

@author: yisu.tian
"""

import os
import re
import pandas as pd
from collections import Counter
from math import log
from sklearn import metrics


# read the file, separate string in emails into words 
# return the numbers of emails and all words in file
def read_file(path):
    files = os.listdir(path)
    s=[]
    for file in files:
        f = open(path+"/"+file,encoding="ISO-8859-1" )
        iter_f = iter(f) 
        str = ""  
        for line in iter_f:  
          str = str + line  
        s+=re.findall(r'\w+',str)
    return (len(files),s)   


# multinomial Naive Bayes training,
# return all words(no dumplicated), p(y) and all conditional probability  matrix p(x|y)
def trainMul(path,stopwords):
    N0 = read_file(path[0])[0]  # number of ham
    N1 = read_file(path[1])[0]  # number of spam
    prior =  [N0/(N0+N1), N1/(N0+N1)]
    set0 = read_file(path[0])[1]  # ham words
    set1 = read_file(path[1])[1]   # spam words
    if stopwords=="Yes":  #  not include the stopwords
        set0=[x for x in set0 if x not in sw_list]
        set1=[x for x in set1 if x not in sw_list]   
    V = list(set(set0 + set1))
    count0 = Counter(set0)  # word count for ham
    count1 = Counter(set1) # word count for spam
    Count_matrix = pd.DataFrame(index= V,columns=['ham', 'spam'])
    deno = [len(V)+len(set0), len(V)+len(set1)]  #denominator
    for word in V:
        Count_matrix.loc[ word,:] = [count0[word], count1[word]]
        
    Condi_matrix = pd.DataFrame(index= V,columns=['ham', 'spam'])
    # conditional probability matrix p(x|y) 
    for word in V:
        Condi_matrix.loc[ word,:] = [(Count_matrix.loc[word,'ham']+1)/deno[0],
                    (Count_matrix.loc[word,'spam']+1)/deno[1]]

    return (V, prior, Condi_matrix)

# apply the multinomial Naive Bayes
# taking words in each test-set mail into function
# return the classification of each mail
def applyMul(test, V, prior, Condi_matrix):
    score_ham, score_spam = log(prior[0]), log(prior[1])
    for word in test:
        if word in V:
            score_ham+=log(Condi_matrix.loc[word,'ham'])
            score_spam+=log(Condi_matrix.loc[word,'spam'])
    if score_ham>score_spam:
        return 0
    else:
        return 1
    
# testing the test set according to  P(Xi|Y), P(Y) from train set
def testing(path_train, path_test,stopwords):
    V=trainMul(path_train,stopwords)[0]
    prior=trainMul(path_train,stopwords)[1]
    Condi_matrix=trainMul(path_train,stopwords)[2]
    
       
    # test ham
    result_ham=[]
    files0=os.listdir(path_test[0])
    for file in files0:
        s=[]
        f = open(path_test[0]+"/"+file,encoding="ISO-8859-1" )
        iter_f = iter(f) 
        str = ""  
        for line in iter_f:  
          str = str + line  
        s+=re.findall(r'\w+',str)
        # taking the words into applyMul function,and store the result to result_ham list
        result_ham.append(applyMul(s, V, prior, Condi_matrix))
    
    # test spam
    result_spam=[]
    files1=os.listdir(path_test[1])
    for file in files1:
        s=[]
        f = open(path_test[1]+"/"+file,encoding="ISO-8859-1" )
        iter_f = iter(f) 
        str = ""  
        for line in iter_f:  
          str = str + line  
        s+=re.findall(r'\w+',str)
        # taking the words into applyMul function,and store the result to result_spam list
        result_spam.append(applyMul(s, V, prior, Condi_matrix))
    
    
    
    right=[0]*len(result_ham)+[1]*len(result_spam)
    # right classification will be used to compare next
    
    result=metrics.accuracy_score( right, result_ham + result_spam )
    return (result)


#########################Stopwords#########################################


sw_list=pd.read_csv("D:/stopwords.txt",header=None)[0].tolist()
# list contains all stopwords    

    
    
##########################test#############################################    
    


path_train = ["D:/train/ham","D:/train/spam"]
path_test = ["D:/test/ham","D:/test/spam"]

testing(path_train, path_test, "Yes")   # delete stopwords
testing(path_train, path_test, "No")    # not delete



