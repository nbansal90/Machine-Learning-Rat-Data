
# coding: utf-8

# In[82]:



##Required libraries###
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.stats import itemfreq
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import svm
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn as skl
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA


# In[2]:

data_read=pd.read_csv("Rats_interp.csv")
data_read.head()
msk = np.random.rand(len(data_read)) < 0.6


# In[3]:

train_data=data_read[msk]
test_data=data_read[~msk]
train_data.head()


# In[4]:

class_genotype=data_read['Genotype']
class_Treatement=data_read['Treatment']
class_Behaviour=data_read['Behavior']
class_class=data_read['class']


# In[5]:

####For training data##
mu = 0
sig = 0.05
train_data_inp=train_data.ix[:,2:79] #Selecting only the input features
train_data_inp=train_data_inp+np.random.normal(mu,sig,1)


# In[6]:

train_genotype=train_data['Genotype']
train_Treatment=train_data['Treatment']
train_Behaviour=train_data['Behavior']
train_class=train_data['class']


# In[7]:

####For testing data##
test_data_inp=test_data.ix[:,2:79] #Selecting only the input features
test_data_inp=test_data_inp+np.random.normal(mu,sig,1)
test_genotype=test_data['Genotype']
test_Treatment=test_data['Treatment']
test_Behaviour=test_data['Behavior']
test_class=test_data['class']


# In[19]:

feature_names=list(train_data_inp)
feature_selection_for_training_data = SelectKBest(score_func=f_classif, k=10)
fit_genotype = feature_selection_for_training_data.fit(train_data_inp, train_genotype)
feature_values_train_genotype = fit_genotype.transform(train_data_inp)
feature_values_test_genotype = fit_genotype.transform(test_data_inp)
       


# In[21]:

mask = feature_selection_for_training_data.get_support() #list of booleans
new_features = [] # The list of your K best features
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)
feature_values_train_genotype=pd.DataFrame(feature_values_train_genotype,columns=new_features)
feature_values_test_genotype=pd.DataFrame(feature_values_test_genotype,columns=new_features)


# In[22]:

fit_Treatment = feature_selection_for_training_data.fit(train_data_inp, train_Treatment)
feature_values_train_Treatment = fit_Treatment.transform(train_data_inp)
feature_values_test_Treatment = fit_Treatment.transform(test_data_inp)
mask = feature_selection_for_training_data.get_support() #list of booleans
new_features = [] # The list of your K best features
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)
feature_values_train_Treatment=pd.DataFrame(feature_values_train_Treatment,columns=new_features)
feature_values_test_Treatment=pd.DataFrame(feature_values_test_Treatment,columns=new_features)


# In[24]:

fit_Behaviour = feature_selection_for_training_data.fit(train_data_inp, train_Behaviour)
feature_values_train_Behaviour = fit_Behaviour.transform(train_data_inp)
feature_values_test_Behaviour = fit_Behaviour.transform(test_data_inp)
mask = feature_selection_for_training_data.get_support() #list of booleans
new_features = [] # The list of your K best features
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)
feature_values_train_Behaviour=pd.DataFrame(feature_values_train_Behaviour,columns=new_features)
feature_values_test_Behaviour=pd.DataFrame(feature_values_test_Behaviour,columns=new_features)


# In[25]:

fit_class = feature_selection_for_training_data.fit(train_data_inp, train_class)
feature_values_train_class = fit_class.transform(train_data_inp)
feature_values_test_class = fit_class.transform(test_data_inp)
mask = feature_selection_for_training_data.get_support() #list of booleans
new_features = [] # The list of your K best features
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)
feature_values_train_class=pd.DataFrame(feature_values_train_class,columns=new_features)
feature_values_test_class=pd.DataFrame(feature_values_test_class,columns=new_features)


# In[26]:

train_data_inp_genotype=feature_values_train_genotype
test_data_inp_genotype=feature_values_test_genotype
        
train_data_inp_Treatment=feature_values_train_Treatment
test_data_inp_Treatment=feature_values_test_Treatment
        
train_data_inp_Behaviour=feature_values_train_Behaviour
test_data_inp_Behaviour=feature_values_test_Behaviour
        
train_data_inp_class=feature_values_train_class
test_data_inp_class=feature_values_test_class


# In[27]:

LDA_genotype=LDA() #For classification of genotype
LDA_Treatment=LDA() #For classification of Treatment
LDA_Behaviour=LDA() #For classification of Behaviour
LDA_class=LDA()
    
LDA_genotype.fit(train_data_inp_genotype,train_genotype)
LDA_genotype_pred=LDA_genotype.predict(test_data_inp_genotype)
LDA_genotype_conf_matrix=confusion_matrix(LDA_genotype_pred,test_genotype)
    
LDA_Treatment.fit(train_data_inp_Treatment,train_Treatment)
LDA_Treatment_pred=LDA_Treatment.predict(test_data_inp_Treatment)
LDA_Treatment_conf_matrix=confusion_matrix(LDA_Treatment_pred,test_Treatment)
    
LDA_Behaviour.fit(train_data_inp_Behaviour,train_Behaviour)
LDA_Behaviour_pred=LDA_Behaviour.predict(test_data_inp_Behaviour)
LDA_Behaviour_conf_matrix=confusion_matrix(LDA_Behaviour_pred,test_Behaviour)
    
LDA_class.fit(train_data_inp_class,train_class)
LDA_class_pred=LDA_class.predict(test_data_inp_class)
LDA_class_conf_matrix=confusion_matrix(LDA_class_pred,test_class)
    
print(LDA_genotype_conf_matrix)
print(LDA_Treatment_conf_matrix)
print(LDA_Behaviour_conf_matrix)
print(LDA_class_conf_matrix)
    
print("               ")
print("Genotype-error",(LDA_genotype_conf_matrix[0][1]+LDA_genotype_conf_matrix[1][0])/ (sum(sum(LDA_genotype_conf_matrix))))
print("Treatment-error",(LDA_Treatment_conf_matrix[0][1]+LDA_Treatment_conf_matrix[1][0])/ (sum(sum(LDA_Treatment_conf_matrix))))
print("Behaviour-error",(LDA_Behaviour_conf_matrix[0][1]+LDA_Behaviour_conf_matrix[1][0])/ (sum(sum(LDA_Behaviour_conf_matrix))))
print("class-error",(1-(np.trace(LDA_class_conf_matrix)/ (sum(sum(LDA_class_conf_matrix))))))


# In[36]:

#Working with SVM
SVM_genotype=svm.SVC(kernel='poly', degree=3, C=1.5).fit(train_data_inp_genotype, train_genotype)
SVM_Treatment=svm.SVC(kernel='poly', degree=3, C=1.5).fit(train_data_inp_Treatment,train_Treatment)
SVM_Behaviour=svm.SVC(kernel='poly', degree=3, C=1.5).fit(train_data_inp_Behaviour,train_Behaviour)
SVM_class=svm.SVC(kernel='poly', degree=3, C=1.0).fit(train_data_inp_class,train_class)
    

SVM_genotype_pred=SVM_genotype.predict(test_data_inp_genotype)
SVM_genotype_conf_matrix=confusion_matrix(SVM_genotype_pred,test_genotype)
    

SVM_Treatment_pred=SVM_Treatment.predict(test_data_inp_Treatment)
SVM_Treatment_conf_matrix=confusion_matrix(SVM_Treatment_pred,test_Treatment)
    

SVM_Behaviour_pred=SVM_Behaviour.predict(test_data_inp_Behaviour)
SVM_Behaviour_conf_matrix=confusion_matrix(SVM_Behaviour_pred,test_Behaviour)
    

SVM_class_pred=SVM_class.predict(test_data_inp_class)
SVM_class_conf_matrix=confusion_matrix(SVM_class_pred,test_class)


# In[37]:

print(SVM_genotype_conf_matrix)
print(SVM_Treatment_conf_matrix)
print(SVM_Behaviour_conf_matrix)
print(SVM_class_conf_matrix)


# In[38]:

print("               ")
print("Genotype-error",(SVM_genotype_conf_matrix[0][1]+SVM_genotype_conf_matrix[1][0])/ (sum(sum(SVM_genotype_conf_matrix))))
print("Treatment-error",(SVM_Treatment_conf_matrix[0][1]+SVM_Treatment_conf_matrix[1][0])/ (sum(sum(SVM_Treatment_conf_matrix))))
print("Behaviour-error",(SVM_Behaviour_conf_matrix[0][1]+SVM_Behaviour_conf_matrix[1][0])/ (sum(sum(SVM_Behaviour_conf_matrix))))
print("class-error",(1-(np.trace(SVM_class_conf_matrix)/ (sum(sum(SVM_class_conf_matrix))))))


# In[42]:

#Working with Random Forest
# n_estimators=10
RF_genotype=RF() #For classification of genotype
RF_Treatment=RF() #For classification of Treatment
RF_Behaviour=RF() #For classification of Behaviour
RF_class=RF()     #For Classification of Class


RF_genotype.fit(train_data_inp_genotype,train_genotype)
RF_genotype_pred=RF_genotype.predict(test_data_inp_genotype)
RF_genotype_conf_matrix=confusion_matrix(RF_genotype_pred,test_genotype)

RF_Treatment.fit(train_data_inp_Treatment,train_Treatment)
RF_Treatment_pred=RF_Treatment.predict(test_data_inp_Treatment)
RF_Treatment_conf_matrix=confusion_matrix(RF_Treatment_pred,test_Treatment)

RF_Behaviour.fit(train_data_inp_Behaviour,train_Behaviour)
RF_Behaviour_pred=RF_Behaviour.predict(test_data_inp_Behaviour)
RF_Behaviour_conf_matrix=confusion_matrix(RF_Behaviour_pred,test_Behaviour)

RF_class.fit(train_data_inp_class,train_class)
RF_class_pred=RF_class.predict(test_data_inp_class)
RF_class_conf_matrix=confusion_matrix(RF_class_pred,test_class)

print(RF_genotype_conf_matrix)
print(RF_Treatment_conf_matrix)
print(RF_Behaviour_conf_matrix)
print(RF_class_conf_matrix)

print("               ")
print("Genotype-error",(RF_genotype_conf_matrix[0][1]+RF_genotype_conf_matrix[1][0])/ (sum(sum(RF_genotype_conf_matrix))))
print("Treatment-error",(RF_Treatment_conf_matrix[0][1]+RF_Treatment_conf_matrix[1][0])/ (sum(sum(RF_Treatment_conf_matrix))))
print("Behaviour-error",(RF_Behaviour_conf_matrix[0][1]+RF_Behaviour_conf_matrix[1][0])/ (sum(sum(RF_Behaviour_conf_matrix))))
print("class-error",(1-(np.trace(RF_class_conf_matrix)/ (sum(sum(RF_class_conf_matrix))))))


# In[47]:

#standardize variables
Numer_data = data_read.ix[:,2:79]
scaler = skl.preprocessing.StandardScaler()
x_st = scaler.fit_transform(Numer_data)


# In[48]:

#Working with Explatory Techniques
#Calculating Eigen Value and Eigen Vector
eig_vals_x, eig_vecs_x = np.linalg.eig(Numer_data.corr())
print (eig_vals_x)
print (eig_vecs_x)


# In[49]:

#Preparing the eigeon Vectors
e_vec = eig_vecs_x.T
e_vec = e_vec[:2]
e_val = eig_vals_x
e_vec = e_vec.T


# In[73]:

res = np.dot(x_st,e_vec)
## Scatterplot of first 2 PCs.
get_ipython().magic('matplotlib inline')
plt.scatter(res[:,0],res[:,1])
plt.title('Largest Two EigenVector')


# In[105]:

train_genotype_label = []
for i in train_genotype:
    if i == 'Ts65Dn':
        train_genotype_label.append('1')
    else:
        train_genotype_label.append('0')
train_Treatment_label = []
for i in train_Treatment:
    if i == 'Saline':
        train_Treatment_label.append('1')
    else:
        train_Treatment_label.append('0')
        
train_Behaviour_label = []
for i in train_Behaviour:
    if i == 'C/S':
        train_Behaviour_label.append('1')
    else:
        train_Behaviour_label.append('0')
        
train_class_label = []
for i in train_class:
    if i == 't-CS-s':
        train_class_label.append('1')
    else:
        train_class_label.append('0')


# In[72]:

plt.scatter(res[:,0],res[:,1],c=train_genotype_label)
plt.title('Genotype Class')


# In[71]:

plt.scatter(res[:,0],res[:,1],c=train_Treatment_label)
plt.title('Treatment Class')


# In[70]:

plt.scatter(res[:,0],res[:,1],c=train_Behaviour_label)
plt.title('Behaviour Class')


plt.scatter(res[:,0],res[:,1],c=train_class_label)
plt.title('Class')

#Multidimensional Scaling
x_st = scaler.fit_transform(train_data_inp)
similarities = euclidean_distances(x_st)
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(similarities).embedding_

# Rescale the data
pos *= np.sqrt((x_st ** 2).sum()) / np.sqrt((pos ** 2).sum())

# Rotate the data
clf = PCA(n_components=2)
X_true = clf.fit_transform(x_st)

pos = clf.fit_transform(pos)

plt.scatter(pos[:, 0], pos[:, 1],  s=20, lw=0, label='MDS',c = train_Treatment_label)
plt.title('train_Treatment_label')

plt.scatter(pos[:, 0], pos[:, 1],  s=20, lw=0, label='MDS',c = train_Behaviour_label)
plt.title('train_Behaviour_label')

plt.scatter(pos[:, 0], pos[:, 1],  s=20, lw=0, label='MDS',c = train_genotype_label)
plt.title('train_genotype_label')



plt.scatter(pos[:, 0], pos[:, 1],  s=20, lw=0, label='MDS',c = train_class_label)
plt.title('train_class_label')




