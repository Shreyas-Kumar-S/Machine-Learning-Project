#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import files
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#load data
cd=pd.read_csv("Customers.csv")
print(cd.head())
#checking the size of dataset
print("The size is:",cd.shape)


# In[2]:


#finding wcss -within cluster sum of square 
x = cd.copy()

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    
    wcss.append(kmeans.inertia_)
#plot an elbow graph
sns.set()
plt.plot(range(1,11),wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.show()


# In[24]:


#Training the K-means clustering model
X2 = cd[['Annual Income (k$)' , 'Spending Score (1-100)']].iloc[: , :].values

algorithm = (KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X2)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_

h = 0.02
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z2 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z2 = Z2.reshape(xx.shape)
plt.imshow(Z2 , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = cd , c = labels2 , 
            s = 200 )
plt.scatter(x = centroids2[: , 0] , y =  centroids2[: , 1] , s = 300 , c = 'Blue' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Annual Income (k$)')
plt.show()
    


# In[5]:


X=cd.iloc[:,1:-1]
y=cd.iloc[:,-1]
print(X.head())
print(y.head())


# In[6]:


#standardize features
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler=StandardScaler()
X=scaler.fit_transform(X)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score,classification_report

# train a Decision tree model on the reduced features and target
dt=LogisticRegression()
dt.fit(X_train,y_train)

# evaluate the accuracy of the model on the testing set

y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# In[8]:


print("\n",classification_report(y_test, dt.predict(X_test)))


# In[15]:


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y, dt.predict(X)))

cm = confusion_matrix(y, dt.predict(X))

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


# In[16]:


def predict(a,b,c,d):
    l=[np.array([a,b,c,d])]
    d=pd.DataFrame(l)
    result= dt.predict(d)
    
    if result==1:
        print("Worthy for Credit Card")
    else:
        print("Not Worthy for Credit Card")
    
predict(1,50,26,6)


# In[ ]:




