Aim of the project:
The aim of this project is to implement and evaluate Naive Bayes and Logistic Regression for text
classification. 

Packages in my project:
1) pandas: read data from data files
2) Counter: count the numbers of words in emails
3) math: calculate exp function
4) metrics: calculate the accuracy
5) os: read the emails from file
6) re: read the emails from file

Instructions to execute this project:
1) This program can be executed within Python 3.6 
2) Open Command Prompt(Windows)/Terminal(Mac OS) and navigate to the current location of this project folder.
3) To run A2-LR program, the path of train and test files and 4 parameters need to be used (parameters: Lambda, Eta, iteration and stopwords="Yes" or "No" to represent delete stopwords or not).
   The sample of test command is: A2-LR.py <training-set>  <test-set> <Lambda> <Eta> <iteration> <stopwords>.     Because training_set contains two files: ham and spam, to load all training_set successfully, we need to set <training-set> path as a list: e.g. <training-set>=["D:/train/ham","D:/train/spam"]. Same as <test-set>:
<test-set>=["D:/test/ham","D:/test/spam"]. 
4) To run A2-NB program, the path of train and test files and one parameter (used to represent deleting stopwords or not) need to be used .
   The sample of test command is: A2-NB.py <training-set> <test-set> <stopwords>. The <training-set> and <test-set> should take list like above.