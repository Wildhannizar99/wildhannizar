# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:44:15 2019

@author: Wildhan Nizar
"""

import pandas as pd

 # Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split function
from sklearn.model_selection import train_test_split 
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 


# load dataset
data = pd.read_csv(r'D:\Keem Hamasah\GEAR 5\PYTHON\PROJEK AKHIR\student-mat.csv')
data.head()

'''-------RECODE DATA-------'''

def nom(series):
    if series == 'yes':
        return '0'
    elif series == 'no':
        return '1'
    else :
        return series

data['Tschoolsup'] = data['schoolsup'].apply(nom)
data['Tfamsup'] = data['famsup'].apply(nom)
data['Tpaid'] = data['paid'].apply(nom)
data['Tactivities'] = data['activities'].apply(nom)
data['Tnursery'] = data['nursery'].apply(nom)
data['Thigher'] = data['higher'].apply(nom)
data['Tinternet'] = data['internet'].apply(nom)

data['y'] = data['Dalc'] + data ['Walc'] 

import statistics 
x = statistics.mean(data['y'])


data['y'] = pd.cut(data['y'],
                     bins=[0,3,11],
                     labels=["0", "1"])


'''-----------------------DATA NOMINAL-----------------------------'''
feature_cols1 = ['Tschoolsup','Tfamsup','Tpaid','Tactivities','Tnursery','Thigher','Tinternet']

A = data[feature_cols1] # Features -- VARIABEL X
b = data.y # Target variable 


#gapminder[['country','year']].head()
#SPLITTING DATA

# Split dataset into training set and test set
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=1) # 70% training and 30% test

'''
0.25 == 0.66

'''
#BUILDING DECISION TREE
# Create Decision Tree classifer object
clf1 = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf1 = clf1.fit(A_train,b_train)

#Predict the response for test dataset
b_pred = clf1.predict(A_test)

#VALUATING MODEL
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(b_test, b_pred))


#classification rate of 67.53%, considered as good accuracy

'''VISUALIZINH DECISION TRESS'''
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


dot_data = StringIO()
export_graphviz(clf1, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols1,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('alc1.png')
Image(graph.create_png())


'''MODEL LOGISTIK'''
