# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 10:26:18 2018

@author: yisu.tian
"""
import pandas as pd
import os
import re
from collections import Counter
from math import exp
from sklearn import metrics



# read all emails together
# return the number of emails and all words in the file
def read_file(path): 
    files = os.listdir(path)
    s=[]
    for file in files:
        f = open(path+"/"+file,encoding="ISO-8859-1" )
        iter_f = iter(f) 
        str = ""  
        for line in iter_f:  # read the email line by line
          str = str + line  
        s+=re.findall(r'\w+',str)
    return (len(files),s)   



# read email one by one 
# return a matrix which each row contains the Counter of words in an email
def read_email(path, V): 
    Count_matrix = pd.DataFrame(columns=V)
    files = os.listdir(path)
    overall=[]
    for file in files:
        s=[]
        f = open(path+"/"+file,encoding="ISO-8859-1" )
        iter_f = iter(f) 
        str = ""  
        for line in iter_f:  
          str = str + line  
        s+=re.findall(r'\w+',str)
        Counter_s = Counter(s) # counter of all words in the email
        row=[]
        for word in V:
            row.append(Counter_s[word])
        overall.append(row)  
    df=pd.DataFrame(overall,columns=V)
    Count_matrix=Count_matrix.append(df,ignore_index=True)
    # put the Counter list into matrix
    return Count_matrix
        
        
# get the Count_matrix and add one Column "Class", if it is ham: Class=0, else 1
def data_set(path,stopwords):
    set0 = read_file(path[0])[1]  # ham words
    set1 = read_file(path[1])[1]   # spam words
    
    if stopwords=="Yes": #  not include the stopwords
        set0=[x for x in set0 if x not in sw_list]
        set1=[x for x in set1 if x not in sw_list] 
        
    V=list(set(set0+set1)) #all words appeared in all mails (no dumplicated)
    # get word count from each ham eamil, add ham_sign=0
    matrix1 = read_email(path[0],V)
    matrix1['Class'] = pd.Series([0]*len(matrix1), index=matrix1.index)
    # get word count from each spam eamil, add spam_sign=1
    matrix2 = read_email(path[1],V)
    matrix2['Class'] = pd.Series([1]*len(matrix2), index=matrix2.index)
    Count_matrix = matrix1.append(matrix2,ignore_index=True)
    return Count_matrix


# calculate the matrix contains all Yl-P(Yl=1|Xl,W)  l: index of email
def inside(w, matrix):
    M=matrix.iloc[:,:-1]*w[:-1]  
    m=M.sum(axis=1)
    new_list=[]
    for i in range(len(m)):
        y=matrix.iloc[i,-1]
        if m[i]+w[-1]>12:
            new_list.append(y-1)
        elif m[i]+w[-1]<-14:
            new_list.append(y)
        else:
            new_list.append(y-(1-1/(1+exp(m[i]+w[-1]))))
    return new_list
    
    

    
# training the train_set emails
# Lambda: weight for weight parameter
# Eta: step size in gradient ascent
# iteration: iteration number in gradient ascent
# return the estimated weight list W   
def get(path, stopwords, Lambda, Eta, iteration):
    Count_matrix=data_set(path,stopwords)
    len_matrix=len(Count_matrix)
    V=list(Count_matrix.dtypes.index)
    len_W=len(V)
    W=[0]*len(V)   # initial value of weight
    itera=0
    while itera<iteration:
        it_W=[]
        new_list=inside(W,Count_matrix) # matrix contains all Yl-P(Yl=1|Xl,W)
        for i in range(len_W-1):
            S=0
            for l in range(len_matrix):
                if Count_matrix.iloc[l,i]==0: # if Xi=0, left part will be 0
                    left=0
                else:
                    y_pro=new_list[l] # find Yl-P(Yl=1|Xl,W) from matrix 
                    left=Count_matrix.iloc[l,i]*y_pro
                S+=left
            S=(S-Lambda*W[i])*Eta
            it_W.append(S+W[i]) # add updated Wi into new weight list
        last=(sum(new_list)-Lambda*W[-1])*Eta    # Weight0 update
        it_W.append(last+W[-1]) # add updated W0 into new weight list
        W=it_W   
        itera=itera+1
    return (V,W)  
 
    
# to classificate the email is ham or spam according to the weight list
def ham_spam(w,data):
    result=sum([x*y for x,y in zip(w,data)])
    if result<0:
        return 0  # if w0+sum(wi*xi)>0, the email would be spam
    else:
        return 1


# testing function include training model and classify all test email
# return the accurancy of test_set email
def testing(path_train, path_test, stopwords,Lambda, Eta, iteration):
    train_result=get(path_train, stopwords, Lambda, Eta, iteration)
    # training train_set emails to get parameter list: Weight
    W=train_result[1]
    V=train_result[0]
        # test ham: get all ham emails from file ham
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
        Counter_s=Counter(s)
        test_mail=[]
        for word in V:
            test_mail.append(Counter_s[word])
        result_ham.append(ham_spam(W,test_mail))
    
    # test spam: get all spam emails from file spam
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
        Counter_s=Counter(s)
        test_mail=[]
        for word in V:
            test_mail.append(Counter_s[word])
        result_spam.append(ham_spam(W,test_mail))
        # return all results
        
    right=[0]*len(result_ham)+[1]*len(result_spam)
    result=metrics.accuracy_score( right, result_ham + result_spam )
    return (result)


#########################Stopwords#########################################

sw_list=pd.read_csv("D:/stopwords.txt",header=None)[0].tolist()

    
##########################test#############################################



path_train = ["D:/train/ham","D:/train/spam"]
path_test = ["D:/test/ham","D:/test/spam"]



testing(path_train, path_test, stopwords="Yes",Lambda=0.001, Eta=0.01, iteration=15)
# delete stopwords

testing(path_train, path_test, stopwords="No",Lambda=0.001, Eta=0.01, iteration=15)
# not delete stopwords

testing(path_train, path_test, stopwords="Yes",Lambda=1, Eta=0.01, iteration=5)
# delete stopwords

testing(path_train, path_test, stopwords="No",Lambda=1, Eta=0.01, iteration=5)
# not delete stopwords


