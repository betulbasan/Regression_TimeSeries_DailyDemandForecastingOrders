# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 20:34:04 2021

@author: betul
"""

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold 
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

#1st Question
# read data
df = pd.read_csv("DailyDemandForecastingOrders.csv", sep=';')#,index_col=[0,1])

df2=df.iloc[5:]
Week = df2['Week'].values.tolist()
Day = df2['Day'].values.tolist()
def scoreResults(model, X_train, X_test, y_train, y_test):

    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)

    r2_train = metrics.r2_score(y_train, y_train_predict)
    r2_test = metrics.r2_score(y_test, y_test_predict)

    mse_train = metrics.mean_squared_error(y_train, y_train_predict)
    mse_test = metrics.mean_squared_error(y_test, y_test_predict)

    return [r2_train, r2_test, mse_train, mse_test]
def residualAnalysis(y_train_pred, y_train, y_test_pred, y_test, title):
    plt.figure(figsize=(12,8))
    plt.scatter(y_train_pred,y_train_pred-y_train,c='blue',marker='o',label="Training Data")
    plt.scatter(y_test_pred,y_test_pred-y_test,c='red',marker='*',label='Test Data')
    plt.title(title)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10,xmax=600, lw=2, color='k')
    plt.xlim([-10, 600])
    plt.show()
def scatterPlot(x,y,y_test,ytd,title):
    axes[x][y].scatter(y_test,ytd,c='blue',marker='o',label='Predicted values for the test set')
    axes[x][y].plot(range(500), range(500),c='green')
    axes[x][y].set_title(title)
    axes[x][y].set_xlabel("Predicted Values")
    axes[x][y].set_ylabel("Residuals")
    axes[x][y].legend(loc='upper left')   
#2nd Question
#Sliding Window for Time Series Preprocessing
dataset=df.values
winsize=5
meanNu=[] #Non-urgent order

for i in range (winsize,len(dataset)):
    meNu=np.mean(dataset[i-winsize:i,2])
    meanNu.append(meNu)
    
plt.subplot(2,1,1)
plt.plot(dataset[5:,2])

plt.subplot(2,1,2)
plt.plot(meanNu)

meanU=[] #Urgent order

for i in range (winsize,len(dataset)):
    meU=np.mean(dataset[i-winsize:i,3])
    meanU.append(meU)
    
meanA=[] #Order type A

for i in range (winsize,len(dataset)):
    meA=np.mean(dataset[i-winsize:i,4])
    meanA.append(meA)
    
meanB=[] #Order type B

for i in range (winsize,len(dataset)):
    meB=np.mean(dataset[i-winsize:i,5])
    meanB.append(meB)
    
meanC=[] #Order type C

for i in range (winsize,len(dataset)):
    meC=np.mean(dataset[i-winsize:i,6])
    meanC.append(meC)

meanFsc=[] #Fiscal sector orders

for i in range (winsize,len(dataset)):
    meFsc=np.mean(dataset[i-winsize:i,7])
    meanFsc.append(meFsc)
    
meanTrf=[] #Traffic controller sector orders

for i in range (winsize,len(dataset)):
    meTrf=np.mean(dataset[i-winsize:i,8])
    meanTrf.append(meTrf)

mean1=[] #Banking orders (1)

for i in range (winsize,len(dataset)):
    me1=np.mean(dataset[i-winsize:i,9])
    mean1.append(me1)

mean2=[] #Banking orders (2)

for i in range (winsize,len(dataset)):
    me2=np.mean(dataset[i-winsize:i,10])
    mean2.append(me2)

mean3=[] #Banking orders (3)

for i in range (winsize,len(dataset)):
    me3=np.mean(dataset[i-winsize:i,11])
    mean3.append(me3)

meanT=[] #Target (Total orders)

for i in range (winsize,len(dataset)):
    meT=np.mean(dataset[i-winsize:i,12])
    meanT.append(meT)
#Sliding variables import new data frame
data_frame = pd.DataFrame(Week,columns=['Week'])
data_frame['Day'] = pd.Series(Day, index=data_frame.index)
data_frame['Non-urgent order'] = pd.Series(meanNu, index=data_frame.index)
data_frame['Urgent order'] = pd.Series(meanU, index=data_frame.index)
data_frame['Order type A'] = pd.Series(meanA, index=data_frame.index)
data_frame['Order type B'] = pd.Series(meanB, index=data_frame.index)
data_frame['Order type C'] = pd.Series(meanC, index=data_frame.index)
data_frame['Fiscal sector orders'] = pd.Series(meanFsc, index=data_frame.index)
data_frame['Traffic controller sector orders'] = pd.Series(meanTrf, index=data_frame.index)
data_frame['Banking orders (1)'] = pd.Series(mean1, index=data_frame.index)
data_frame['Banking orders (2)'] = pd.Series(mean2, index=data_frame.index)
data_frame['Banking orders (3)'] = pd.Series(mean3, index=data_frame.index)
data_frame['Target (Total orders)'] = pd.Series(meanT, index=data_frame.index)

# standardization
from sklearn.preprocessing import StandardScaler
print(data_frame.head())
# define standard scaler
scaler = StandardScaler()
# transform data
scaler.fit_transform(data_frame)

#3rd Question
X=data_frame.iloc[:,:-1].values
y=data_frame.iloc[:,-1].values

#Multiple Regressions with sklearn 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
ytd1=lr.predict(X_test)

# Mean Squared Error (MSE)
print('MSE for Linear regression (train) is:' ,mean_squared_error(y_train, y_train_pred))
print('MSE for Linear regression (test) is:' ,mean_squared_error(y_test, y_test_pred))
# Coefficient of Determination,  R2
print('R^2 for Linear regression (train) is:', r2_score(y_train, y_train_pred))
print('R^2 for Linear regression (test) is:' ,r2_score(y_test, y_test_pred))
residualAnalysis(y_train_pred, y_train, y_test_pred, y_test, "Linear Regression Algorithm" )

#Ridge regression algorithm
rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train) 
pred_train_rr= rr.predict(X_train)
pred_test_rr=rr.predict(X_test)
ytd2=rr.predict(X_test)

#Mean Squared Error(MSE)
print('MSE for Ridge regression (train) is:', mean_squared_error(y_train, y_train_pred))
print('MSE for Ridge regression (test) is:', mean_squared_error(y_test, y_test_pred))
#Coefficient of Determination R^2
print('R^2 for Ridge regression (train) is:', r2_score(y_train, y_train_pred))
print('R^2 for Ridge regression (test) is:', r2_score(y_test,y_test_pred))
residualAnalysis(pred_train_rr, y_train, pred_test_rr, y_test, "Ridge Regression Algorithm")

#Lasso Regression Algorithm
lasso_model = Lasso(alpha=1.0)
lasso=lasso_model.fit(X_train , y_train)
pred_train_lasso=lasso_model.predict(X_train)
pred_test_lasso=lasso_model.predict(X_test)
ytd3=rr.predict(X_test)

#Mean Squared Error(MSE)
print('MSE for Lasso regression (train) is:', mean_squared_error(y_train,y_train_pred))
print('MSE for Lasso regression (test) is:', mean_squared_error(y_test, y_test_pred))
#Coefficient of Determination R^2
print('R^2 for Lasso regression (train) is:', r2_score(y_train, y_train_pred))
print('R^2 for Lasso regression (test) is:', r2_score(y_test,y_test_pred))
residualAnalysis(pred_train_rr, y_train, pred_test_rr, y_test, "Lasso Regression Algorithm")

#ElasticNet Regression Algorithm
elastic_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic=elastic_model.fit(X_train,y_train)
pred_train_elastic=elastic_model.predict(X_train)
pred_test_elastic=elastic_model.predict(X_test)
ytd4=rr.predict(X_test)

#Mean Squared Error(MSE)
print('MSE for ElasticNet regression (train) is:', mean_squared_error(y_train,y_train_pred))
print('MSE for ElasticNet regression (test) is:', mean_squared_error(y_test, y_test_pred))
#Coefficient of Determination R^2
print('R^2 for ElasticNet regression (train) is:', r2_score(y_train,y_train_pred))
print('R^2 for ElasticNet regression (test) is:',r2_score(y_test, y_test_pred))
residualAnalysis(pred_train_elastic, y_train, pred_test_elastic, y_test, "ElasticNet Regression Algorithm")

#K-NN Regression Algorithm
knn_model = KNeighborsRegressor(n_neighbors=5)
knn=knn_model.fit(X_train, y_train)
pred_train_knn=knn_model.predict(X_train)
pred_test_knn=knn_model.predict(X_test)
ytd5=rr.predict(X_test)

#Mean Squared Error(MSE)
print('MSE for K-NN regression (train) is:', mean_squared_error(y_train,y_train_pred))
print('MSE for K-NN regression (test) is:',mean_squared_error(y_test, y_test_pred))
#Coefficient of Determination R^2
print('R^2 for K-NN regression (train) is:',r2_score(y_train,y_train_pred))
print('R^2 for K-NN regression (test) is:',r2_score(y_test, y_test_pred))
residualAnalysis(pred_train_knn, y_train, pred_test_knn, y_test, "K-NN Regression Algorithm")

#Decision Tree Algorithm
decision_model = DecisionTreeRegressor()
decision=decision_model.fit(X_train, y_train)
pred_train_dec=decision_model.predict(X_train)
pred_test_dec=decision_model.predict(X_test)
ytd6=rr.predict(X_test)

#Mean Squared Error(MSE)
print('MSE for Decision Tree (train) is:',mean_squared_error(y_train,y_train_pred))
print('MSE for Decision Tree (test) is:',mean_squared_error(y_test, y_test_pred))
#Coefficient of Determination R^2
print('R^2 for Decision Tree (train) is:', r2_score(y_train,y_train_pred))
print('R^2 for Decision Tree (test) is:', r2_score(y_test, y_test_pred))
residualAnalysis(pred_train_knn, y_train, pred_test_knn, y_test, "Decision Tree Regression Algorithm")

#Random Forest Algorithm
random_model=RandomForestRegressor()
random=random_model.fit(X_train, y_train)
pred_train_ran=random_model.predict(X_train)
pred_test_ran=random_model.predict(X_test)
ytd7=rr.predict(X_test)

#Mean Squared Error(MSE)
print('MSE for Random Forest (train) is:', mean_squared_error(y_train,y_train_pred))
print('MSE for Random Forest (test) is:', mean_squared_error(y_test, y_test_pred))
#Coefficient of Determination R^2
print('R^2 for Random Forest (train) is:', r2_score(y_train,y_train_pred))
print('R^2 for Random Forest(test) is:', r2_score(y_test, y_test_pred))
residualAnalysis(pred_train_ran, y_train, pred_test_ran, y_test, "Random Forest Regression Algorithm")

#Support Vector Regressor Algorithm
support_model = SVR()
support=support_model.fit(X_train,y_train)
pred_train_sup=support_model.predict(X_train)
pred_test_sup=support_model.predict(X_test)
ytd8=rr.predict(X_test)

#Mean Squared Error(MSE)
print('MSE for Support Vector Regressor (train) is:', mean_squared_error(y_train,y_train_pred))
print('MSE for Support Vector Regressor (test) is:', mean_squared_error(y_test, y_test_pred))
#Coefficient of Determination R^2
print('R^2 for Support Vector Regressor (train) is:',r2_score(y_train,y_train_pred))
print('R^2 for Support Vector Regressor (test) is:', r2_score(y_test, y_test_pred))
residualAnalysis(pred_train_sup, y_train, pred_test_sup, y_test, "Support Vector Regressor Algorithm")

#ScatterPlot Matrix
f = plt.figure()    
f, axes = plt.subplots(nrows = 3, ncols = 3, sharex=True, sharey = True)
f.set_size_inches(18.5, 10.5)
scatterPlot(0,0,y_test,ytd1,"Linear Regression Algorithm")
scatterPlot(0,1,y_test,ytd2,"Ridge Regression Algorithm")
scatterPlot(0,2,y_test,ytd3,"Lasso Regression Algorithm")
scatterPlot(1,0,y_test,ytd4,"ElasticNet Regression Algorithm")
scatterPlot(1,1,y_test,ytd5,"K-NN Regression Algorithm")
scatterPlot(1,2,y_test,ytd6,"Decision Tree Regression Algorithm")
scatterPlot(2,0,y_test,ytd7,"Random Forest Regression Algorithm")
scatterPlot(2,1,y_test,ytd8,"Support Vector Regressor Algorithm")

#4th Question
scores = []
lr_cv = LinearRegression()
k = 10
iter = 1
cv = KFold(n_splits=k, random_state = 0, shuffle=True)
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lr_cv.fit(X_train, y_train)

    result = scoreResults(model = lr_cv
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)

    y_train_pred = lr_cv.predict(X_train)
    y_test_pred = lr_cv.predict(X_test)

    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(lr_cv.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Linear Regression Algorithm")

#Ridge Regression Algorithm
ridge_cv = Ridge()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    ridge_cv.fit(X_train, y_train)
    
    result = scoreResults(model = ridge_cv
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = ridge_cv.predict(X_train)
    y_test_pred = ridge_cv.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(ridge_cv.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Ridge Regression Algorithm")
    
#Lasso Regression Algorithm 
lasso_cv = Lasso()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lasso_cv.fit(X_train, y_train)
    
    result = scoreResults(model = lasso_cv
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = lasso_cv.predict(X_train)
    y_test_pred = lasso_cv.predict(X_test)
    
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(lasso_cv.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Lasso Regression Algorithm")
    
#ElasticNet Regression Algorithm
elastic_cv = ElasticNet()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    elastic_cv.fit(X_train, y_train)
    
    result = scoreResults(model = elastic_cv
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = elastic_cv.predict(X_train)
    y_test_pred = elastic_cv.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(elastic_cv.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"ElasticNet Regression Algorithm")
    
#K-NN Regression Algorithm
knn_cv = KNeighborsRegressor()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    knn_cv.fit(X_train, y_train)
    
    result = scoreResults(model = knn_cv
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = knn_cv.predict(X_train)
    y_test_pred = knn_cv.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(knn_cv.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"K-NN Regression Algorithm")
    
#Decision Tree Algorithm
decision_cv = DecisionTreeRegressor()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    decision_cv.fit(X_train, y_train)
    
    result = scoreResults(model = decision_cv
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = decision_cv.predict(X_train)
    y_test_pred = decision_cv.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(decision_cv.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Decision Tree Regression Algorithm")
    
#Random Forest Algorithm
random_cv = RandomForestRegressor()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    random_cv.fit(X_train, y_train)
    
    result = scoreResults(model = random_cv
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = random_cv.predict(X_train)
    y_test_pred = random_cv.predict(X_test)

    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(random_cv.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Random Forest Regression Algorithm")
    
#Support Vector Regressor Algorithm
support_cv = SVR()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    support_cv.fit(X_train, y_train)
    
    result = scoreResults(model = support_cv
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = support_cv.predict(X_train)
    y_test_pred = support_cv.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(support_cv.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Support Vector Regressor Algorithm")

#5th Question
# Correlation Matrix
pd.options.display.float_format = '{:,.2f}'.format
corr_matrix = data_frame.corr()
# elemination/detection of low correlated features
corr_matrix[np.abs(corr_matrix) < 0.6] = 0
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
plt.show()

#Feature with the highest coefficient import new data frame
df3 = pd.DataFrame(Week,columns=['Week'])
df3['Day'] = pd.Series(Day, index=df3.index)
df3['Non-urgent order'] = pd.Series(meanNu, index=df3.index)
df3['Order type A'] = pd.Series(meanA, index=df3.index)
df3['Order type B'] = pd.Series(meanB, index=df3.index)
df3['Banking orders (1)'] = pd.Series(mean1, index=df3.index)
df3['Banking orders (2)'] = pd.Series(mean2, index=df3.index)
df3['Target (Total orders)'] = pd.Series(meanT, index=df3.index)

X=df3.iloc[:,:-1].values
y=df3.iloc[:,-1].values
#Linear Regression Algorithm
scores = []
lr_cv2 = LinearRegression()
k = 10
iter = 1
cv = KFold(n_splits=k, random_state = 0, shuffle=True)
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lr_cv2.fit(X_train, y_train)

    result = scoreResults(model = lr_cv2
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = lr_cv2.predict(X_train)
    y_test_pred = lr_cv2.predict(X_test)


    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(lr_cv2.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Linear Regression Algorithm")

#Ridge Regression Algorithm
rr_cv2 = Ridge()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    rr_cv2.fit(X_train, y_train)
    
    result = scoreResults(model = rr_cv2
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = rr_cv2.predict(X_train)
    y_test_pred = rr_cv2.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(rr_cv2.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Ridge Regression Algorithm")
    
#Lasso Regression Algorithm 
lasso_cv2 = Lasso()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lasso_cv2.fit(X_train, y_train)
    
    result = scoreResults(model = lasso_cv2
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = lasso_cv2.predict(X_train)
    y_test_pred = lasso_cv2.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(lasso_cv2.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Lasso Regression Algorithm")
    
#ElasticNet Regression Algorithm
elastic_cv2 = ElasticNet(max_iter=4000)
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    elastic_cv2.fit(X_train, y_train)
    
    result = scoreResults(model = elastic_cv2
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = elastic_cv2.predict(X_train)
    y_test_pred = elastic_cv2.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(elastic_cv2.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"ElasticNet Regression Algorithm")
    
#K-NN Regression Algorithm
knn_cv2 = KNeighborsRegressor()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    knn_cv2.fit(X_train, y_train)
    
    result = scoreResults(model = knn_cv2
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = knn_cv2.predict(X_train)
    y_test_pred = knn_cv2.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(knn_cv2.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"K-NN Regression Algorithm")

#Decision Tree Algorithm
decision_cv2 = DecisionTreeRegressor()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    decision_cv2.fit(X_train, y_train)
    
    result = scoreResults(model = decision_cv2
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = decision_cv2.predict(X_train)
    y_test_pred = decision_cv2.predict(X_test)
    
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(decision_cv2.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Decision Tree Regression Algorithm") 
    
#Random Forest Algorithm
random_cv2 = RandomForestRegressor()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    random_cv2.fit(X_train, y_train)
    
    result = scoreResults(model = random_cv2
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = random_cv2.predict(X_train)
    y_test_pred = random_cv2.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(random_cv2.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Random Forest Regression Algorithm")
    
#Support Vector Regressor Algorithm
support_cv2 = SVR()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    support_cv2.fit(X_train, y_train)
    
    result = scoreResults(model = support_cv2
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = support_cv2.predict(X_train)
    y_test_pred = support_cv2.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(support_cv2.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Support Vector Regression Algorithm")

#6th Question
#Week and Target (Total orders)
X = data_frame.iloc[:, 0:1].values
y=data_frame.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr_week = LinearRegression()
lr_week.fit(X_train, y_train)
y_train_pred = lr_week.predict(X_train)
y_test_pred = lr_week.predict(X_test)

plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('Week')
plt.ylabel("Total (Target oders)")
plt.show();

sns.jointplot(x='Week', y='Target (Total orders)', data=data_frame, kind='reg', height=8);
plt.show();

# Mean Squared Error (MSE)
print('MSE for Linear regression for week (train) is:', mean_squared_error(y_train, y_train_pred))
print('MSE for Linear regression for week (test) is:', mean_squared_error(y_test, y_test_pred))

#Day and Target (Total orders)
X = data_frame.iloc[:, 1:2].values
y=data_frame.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr_day = LinearRegression()
lr_day.fit(X_train, y_train)
y_train_pred = lr_day.predict(X_train)
y_test_pred = lr_day.predict(X_test)

plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('Day')
plt.ylabel("Target (Total orders)")
plt.show();

sns.jointplot(x='Day', y='Target (Total orders)', data=data_frame, kind='reg', height=8);
plt.show();

# Mean Squared Error (MSE)
print('MSE for Linear regression for day (train) is:', mean_squared_error(y_train, y_train_pred))
print('MSE for Linear regression for day (test) is:', mean_squared_error(y_test, y_test_pred))

# Non-urgent order and Target (Total orders)
X = data_frame.iloc[:, 2:3].values
y=data_frame.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr_nu = LinearRegression()
lr_nu.fit(X_train, y_train)
y_train_pred = lr_nu.predict(X_train)
y_test_pred = lr_nu.predict(X_test)

plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('Non-urgent order')
plt.ylabel("Target (Total orders)")
plt.show();

sns.jointplot(x='Non-urgent order', y='Target (Total orders)', data=data_frame, kind='reg', height=8);
plt.show();

# Mean Squared Error (MSE)
print('MSE for Linear regression for non-urgent order (train) is:', mean_squared_error(y_train, y_train_pred))
print('MSE for Linear regression for non-urgent order (test) is:', mean_squared_error(y_test, y_test_pred))

#Urgent order and Target (Total orders)
X = data_frame.iloc[:, 3:4].values
y=data_frame.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr_orda = LinearRegression()
lr_orda.fit(X_train, y_train)
y_train_pred = lr_orda.predict(X_train)
y_test_pred = lr_orda.predict(X_test)

plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('Urgent order')
plt.ylabel("Target (Total orders)")
plt.show();

sns.jointplot(x='Urgent order', y='Target (Total orders)', data=data_frame, kind='reg', height=8);
plt.show();

# Mean Squared Error (MSE)
print('MSE for Linear regression for urgent order (train) is:', mean_squared_error(y_train, y_train_pred))
print('MSE for Linear regression for urgent order (test) is:', mean_squared_error(y_test, y_test_pred))

#Order type A and Target (Total orders)
X = data_frame.iloc[:, 4:5].values
y=data_frame.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr_orda = LinearRegression()
lr_orda.fit(X_train, y_train)
y_train_pred = lr_orda.predict(X_train)
y_test_pred = lr_orda.predict(X_test)

plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('Order type A')
plt.ylabel("Target (Total orders)")
plt.show();

sns.jointplot(x='Order type A', y='Target (Total orders)', data=data_frame, kind='reg', height=8);
plt.show();

# Mean Squared Error (MSE)
print('MSE for Linear regression for order type A (train) is:', mean_squared_error(y_train, y_train_pred))
print('MSE for Linear regression for order type A (test) is:', mean_squared_error(y_test, y_test_pred))

# Order type B and Target (Total orders)
X = data_frame.iloc[:, 5:6].values
y=data_frame.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr_ordb = LinearRegression()
lr_ordb.fit(X_train, y_train)
y_train_pred = lr_ordb.predict(X_train)
y_test_pred = lr_ordb.predict(X_test)

plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('Order type B')
plt.ylabel("Target (Total orders)")
plt.show();

sns.jointplot(x='Order type B', y='Target (Total orders)', data=data_frame, kind='reg', height=8);
plt.show();

# Mean Squared Error (MSE)
print('MSE for Linear regression for order type B (train) is:', mean_squared_error(y_train, y_train_pred))
print('MSE for Linear regression for order type B (test) is:', mean_squared_error(y_test, y_test_pred))

# Order type C and Target (Total orders)
X = data_frame.iloc[:, 6:7].values
y=data_frame.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr_ordb = LinearRegression()
lr_ordb.fit(X_train, y_train)
y_train_pred = lr_ordb.predict(X_train)
y_test_pred = lr_ordb.predict(X_test)

plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('Order type C')
plt.ylabel("Target (Total orders)")
plt.show();

sns.jointplot(x='Order type C', y='Target (Total orders)', data=data_frame, kind='reg', height=8);
plt.show();

# Mean Squared Error (MSE)
print('MSE for Linear regression for Order type C (train) is:', mean_squared_error(y_train, y_train_pred))
print('MSE for Linear regression for Order type C (test) is:', mean_squared_error(y_test, y_test_pred))

# Fiscal sector orders and Target (Total orders)
X = data_frame.iloc[:, 7:8].values
y=data_frame.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr_ordb = LinearRegression()
lr_ordb.fit(X_train, y_train)
y_train_pred = lr_ordb.predict(X_train)
y_test_pred = lr_ordb.predict(X_test)

plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('Fiscal sector orders')
plt.ylabel("Target (Total orders)")
plt.show();

sns.jointplot(x='Fiscal sector orders', y='Target (Total orders)', data=data_frame, kind='reg', height=8);
plt.show();

# Mean Squared Error (MSE)
print('MSE for Linear regression for Fiscal sector orders (train) is:', mean_squared_error(y_train, y_train_pred))
print('MSE for Linear regression for Fiscal sector orders (test) is:', mean_squared_error(y_test, y_test_pred))

# Traffic controller sector orders and Target (Total orders)
X = data_frame.iloc[:, 8:9].values
y=data_frame.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr_ordb = LinearRegression()
lr_ordb.fit(X_train, y_train)
y_train_pred = lr_ordb.predict(X_train)
y_test_pred = lr_ordb.predict(X_test)

plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('Traffic controller sector orders')
plt.ylabel("Target (Total orders)")
plt.show();

sns.jointplot(x='Traffic controller sector orders', y='Target (Total orders)', data=data_frame, kind='reg', height=8);
plt.show();

# Mean Squared Error (MSE)
print('MSE for Linear regression for Traffic controller sector orders (train) is:', mean_squared_error(y_train, y_train_pred))
print('MSE for Linear regression for Traffic controller sector orders (test) is:', mean_squared_error(y_test, y_test_pred))

#Banking Orders (1) and Target (Total orders)
X = data_frame.iloc[:, 9:10].values
y=data_frame.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr_bank1 = LinearRegression()
lr_bank1.fit(X_train, y_train)
y_train_pred = lr_bank1.predict(X_train)
y_test_pred = lr_bank1.predict(X_test)

plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('Banking orders (1)')
plt.ylabel("Target (Total orders)")
plt.show();

sns.jointplot(x='Banking orders (1)', y='Target (Total orders)', data=data_frame, kind='reg', height=8);
plt.show();

# Mean Squared Error (MSE)
print('MSE for Linear regression for banking orders (1) (train) is:', mean_squared_error(y_train, y_train_pred))
print('MSE for Linear regression for banking orders (1) (test) is:', mean_squared_error(y_test, y_test_pred))

#Banking Orders (2) and Target (Total orders)
X = data_frame.iloc[:, 10:11].values
y=data_frame.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr_bank2 = LinearRegression()
lr_bank2.fit(X_train, y_train)
y_train_pred = lr_bank2.predict(X_train)
y_test_pred = lr_bank2.predict(X_test)

plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('Banking Orders (2)')
plt.ylabel("Target (Total orders)")
plt.show();

sns.jointplot(x='Banking orders (2)', y='Target (Total orders)', data=data_frame, kind='reg', height=8);
plt.show();

# Mean Squared Error (MSE)
print('MSE for Linear regression for banking orders (2) (train) is:', mean_squared_error(y_train, y_train_pred))
print('MSE for Linear regression for banking orders (2) (test) is:', mean_squared_error(y_test, y_test_pred))

#Banking Orders (3) and Target (Total orders)
X = data_frame.iloc[:, 11:12].values
y=data_frame.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr_bank2 = LinearRegression()
lr_bank2.fit(X_train, y_train)
y_train_pred = lr_bank2.predict(X_train)
y_test_pred = lr_bank2.predict(X_test)

plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('Banking Orders (3)')
plt.ylabel("Target (Total orders)")
plt.show();

sns.jointplot(x='Banking orders (3)', y='Target (Total orders)', data=data_frame, kind='reg', height=8);
plt.show();

# Mean Squared Error (MSE)
print('MSE for Linear regression for banking orders (3) (train) is:', mean_squared_error(y_train, y_train_pred))
print('MSE for Linear regression for banking orders (3) (test) is:', mean_squared_error(y_test, y_test_pred))

non_urgent_order= data_frame['Non-urgent order'].values.tolist()
df4 = pd.DataFrame(non_urgent_order,columns=['Non-urgent order'])
df4['Urgent order'] = pd.Series(data_frame['Urgent order'], index=data_frame.index)
df4['Order type B'] = pd.Series(data_frame['Order type B'], index=data_frame.index)
df4['Order type C'] = pd.Series(data_frame['Order type C'], index=data_frame.index)
df4['Banking orders (1)'] = pd.Series(data_frame['Banking orders (1)'], index=data_frame.index)
df4['Banking orders (2)'] = pd.Series(data_frame['Banking orders (2)'], index=data_frame.index)
df4['Target (Total orders)'] = pd.Series(data_frame['Target (Total orders)'], index=data_frame.index)

X=df4.iloc[:,:-1].values
y=df4.iloc[:,-1].values
#Linear Regression Algorithm
lr_cv3 = LinearRegression()
k = 10
iter = 1
cv = KFold(n_splits=k, random_state = 0, shuffle=True)
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lr_cv3.fit(X_train, y_train)

    result = scoreResults(model = lr_cv3
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = lr_cv3.predict(X_train)
    y_test_pred = lr_cv3.predict(X_test)

    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(lr_cv3.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Linear Regression Algorithm")
    
#for Ridge Regression Algorithm
rr_cv3 = Ridge()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    rr_cv3.fit(X_train, y_train)
    
    result = scoreResults(model = rr_cv3
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = rr_cv3.predict(X_train)
    y_test_pred = rr_cv3.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(rr_cv3.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Ridge Regression Algorithm")
    
#Lasso Regression Algorithm 
lasso_cv3 = Lasso()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lasso_cv3.fit(X_train, y_train)
    
    result = scoreResults(model = lasso_cv3
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = lasso_cv3.predict(X_train)
    y_test_pred = lasso_cv3.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(lasso_cv3.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Lasso Regression Algorithm")
    
#ElasticNet Regression Algorithm
elastic_cv3 = ElasticNet()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    elastic_cv3.fit(X_train, y_train)
    
    result = scoreResults(model = elastic_cv3
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = elastic_cv3.predict(X_train)
    y_test_pred = elastic_cv3.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(elastic_cv3.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"ElasticNet Regression Algorithm")
    
#K-NN Regression Algorithm
knn_cv3 = KNeighborsRegressor()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    knn_cv3.fit(X_train, y_train)
    
    result = scoreResults(model = knn_cv3
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = knn_cv3.predict(X_train)
    y_test_pred = knn_cv3.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(knn_cv3.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"K-NN Regression Algorithm")
    
#Decision Tree Algorithm
decision_cv3 = DecisionTreeRegressor()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    decision_cv3.fit(X_train, y_train)
    
    result = scoreResults(model = decision_cv3
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = decision_cv3.predict(X_train)
    y_test_pred = decision_cv3.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(decision_cv3.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Decision Tree Regression Algorithm")
    
#Random Forest Algorithm
random_cv3 = RandomForestRegressor()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    random_cv3.fit(X_train, y_train)
    
    result = scoreResults(model = random_cv3
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = random_cv3.predict(X_train)
    y_test_pred = random_cv3.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(random_cv3.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Random Forest Regression Algorithm")
    
#Support Vector Regressor Algorithm
support_cv3 = SVR()
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    support_cv3.fit(X_train, y_train)
    
    result = scoreResults(model = support_cv3
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
    
    y_train_pred = support_cv3.predict(X_train)
    y_test_pred = support_cv3.predict(X_test)
    
    print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1
    scores.append(support_cv3.score(X_test, y_test))
    residualAnalysis(y_train_pred,y_train,y_test_pred,y_test,"Support Vector Regressor Algorithm")




