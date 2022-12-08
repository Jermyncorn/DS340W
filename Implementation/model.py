#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# import package
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc,roc_auc_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from scipy import stats


# read data
df = pd.read_excel('data.xlsx', sheet_name = 'data')
# features
X = df.iloc[:,[1,2,5,6,7,8]]
# output
Y = df['Outcome']
# descriptive analysis
x_desc = X.describe()
# Non diabetes patients
b = df[df['Outcome']== 0]
# diabetes patients
c = df[df['Outcome']== 1]
# Descriptive statistics of diabetes patients
dia_desc = c.describe()
# Descriptive statistics of Non diabetes patients
undia_desc = b.describe()
#print(x_desc)
#print(dia_desc)
#print(undia_desc)

# cloumn names
col_name = list(X.columns)



# T-test
def t_test(df,col):
    # Select row with output 0
    b = df[df['Outcome']== 0]
    # Select row with output 1
    c = df[df['Outcome']== 1]
    # Select columns for t-test
    b = b[col]
    c = c[col]
    # Compare the number of two samples and randomly sample to the same number of samples
    if len(b) >= len(c):
        l = len(c)
    else:
        l = len(b)
    b = b.sample(n = l)
    # Test homogeneity of variance and t-test. If p>=0.05 of homogeneity of variance, set equal_ var=True
    if list(stats.levene(b, c))[1] >= 0.05:
        d = stats.ttest_ind(b,c,equal_var=True)
        e = stats.levene(b, c)
    else:
        d = stats.ttest_ind(b,c,equal_var=False)
        e = stats.levene(b, c)
    return d,e
# T-test with cyclic ergodic feature
for i in range(0,len(col_name)):
    a,b = t_test(df,col_name[i])
    p_val = a[1]
    print(col_name[i],p_val)



# Probability Density Function of Normal Distribution
#   x      A specific measurement in a data set
#   mu     The average value of the data set, reflecting the centralized trend of the distribution of measured values
#   sigma  The standard deviation of the data set reflects the degree of dispersion of the distribution of measured values
def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf

# Normal distribution map
def hist_firgure(col):
    data = X[col] # Get Dataset
    mean = data.mean() # Get the average value of the dataset
    std = data.std()   # Obtain the standard deviation of the dataset
    # Set X axis: the first two numbers are the starting and ending ranges of X axis, and the third number represents the step size
    # The smaller the step size is set, the smoother the normal distribution curve is drawn
    min_val = data.min()
    max_val = data.max()
    x = np.arange(min_val, max_val, 0.1)
    # Set the Y axis and load the normal distribution function just defined
    y = normfun(x, mean, std)
    # Draw the normal distribution curve of the dataset
    plt.plot(x, y)
    # Draw a histogram of a dataset
    plt.hist(data, bins=20, rwidth=0.9, density=True)
    plt.title('distribution')
    plt.xlabel(col)
    plt.ylabel('Probability') 
     # Output normal distribution curve and histogram
    plt.show()
 
'''
# Output histogram
for i in range(0,len(col_name)):
    hist_firgure(col_name[i])
'''

# Standardize continuity data
std = preprocessing.StandardScaler()
std.fit(X)
Z = std.transform(X)
X = pd.DataFrame(Z,columns =list(X.columns))

# Divide training set and test set
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size= 0.3)



'''
# set gridsearch
rf = RandomForestClassifier(class_weight = 'balanced')
distributions = dict(max_depth=list(range(2,20,2)),
                   min_samples_split=list(range(1,10,1)),
                min_samples_leaf = list(range(1,10,1)),
                min_weight_fraction_leaf = [0,0.1,0.2,0.3,0.4,0.5])              
# set RandomizedSearchCV       
model = GridSearchCV(rf,distributions,scoring = 'recall')
model.fit(x_train,y_train)
print(model.best_params_)
print(model.best_score_)
'''



'''
svc = SVC(class_weight = 'balanced',probability=True)
distributions = dict(C = [1,2,3,4,5],
                     kernel = ('linear', 'poly', 'rbf', 'sigmoid'), 
                     degree = list(range(1,5,1)))        
# set RandomizedSearchCV       
model = GridSearchCV(svc,distributions,scoring = 'recall')
model.fit(x_train,y_train)
print(model.best_params_)
print(model.best_score_)
'''


# Fitting model with adjusted parameters
model = RandomForestClassifier(class_weight = 'balanced',max_depth = 2, min_samples_leaf = 1, min_samples_split= 5, min_weight_fraction_leaf = 0)
model.fit(x_train,y_train)


# feature_name
feature_name = model.feature_names_in_
#feature_importance
feature_importance = model.feature_importances_
dict_all = dict(zip(feature_name, feature_importance))
print(dict_all)
result = model.predict(x_test)

# accuracy
model_score = model.score(x_test,y_test)

print(model_score)


# Output confusion matrix
confusion_matrix = confusion_matrix(y_test,result)
print(confusion_matrix)

# output recall
recall = classification_report(y_test,result)
print(recall)

# Value corresponding to confusion matrix
tp = confusion_matrix[0][0] 
fn = confusion_matrix[0][1]
fp = confusion_matrix[1][0]
tn = confusion_matrix[1][1]


# Draw roc curve
auc = roc_auc_score(y_test,model.predict_proba(x_test)[:,1])
# auc = roc_auc_score(y_test,clf.decision_function(X_test))
fpr,tpr, thresholds = roc_curve(y_test,model.predict_proba(x_test)[:,1])
plt.plot(fpr,tpr,color='darkorange',label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()








 
    
 
    
 
    

 
