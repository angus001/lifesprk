#%% Import packages
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from matplotlib import pyplot as plt
import os
import datetime
from sklearn import neighbors
import seaborn as sns

#%% read csv

df_mem = pd.read_csv(".//members.csv")

#%% read admission and scores
df_admin = pd.read_csv(".//admissions.csv")
df_score = pd.read_csv(".//scores.csv")


#%% create a pivot

dfadmin_pivot = df_admin.groupby("id").count()[["admission_date"]]
# reset index
dfadmin_pivot =dfadmin_pivot.reset_index()


#%% sort value

dfadmin_pivot.sort_values(["admission_date"], ascending=[False], inplace = True )



#%% create pivot by admission count

dfpt1 = dfadmin_pivot.groupby("admission_date").count().reset_index()




#%%

# dfmerg = df_admin.merge(df_score, left_on= 'id', right_on = 'id', how = 'inner' )

#%%

# dfmerg.groupby('id').mean()[['SDOH_FOOD_INSECURITY', 'SDOH_CRIME_RATES', 'ACG_CHF_IND_CD']]

#%% merge with score

dfsc_ad = dfadmin_pivot.merge(df_score, left_on= 'id', right_on = 'id', how = 'inner')
# dfsc_ad.head()




#%% rename columns

dfsc_ad.rename(columns = {'admission_date': "admission_count"}, inplace = True)

#%%

# dfsc_ad.head()

#%% create mean & median group
dfsc_admean = dfsc_ad.groupby('admission_count').mean().reset_index() #[["ACG_CHF_IND_CD", "ACG_DEPRESSION_MPR", "ACG_CHRONIC_CONDITION_COUNT"]]
dfsc_admedian = dfsc_ad.groupby('admission_count').median().reset_index()

#%% create correlation plot
#dfsc_admean.head()

corr = dfsc_admean.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

#%% rename columns
dfpt1.rename(columns = { 'admission_date' : 'admission_count', 'id': 'member_count'}, inplace = True)

#%%
#dfpt1.head()
# dfsc_admean

#%% plot features
plt.style.use('fivethirtyeight')
plt.bar(height = dfpt1['member_count'], x=dfpt1['admission_count'])
plt.plot([x**(1.2) for x in dfsc_admean['SDOH_EDUCATION_ACCESS']], 'k:', label = 'Education')
plt.plot(dfsc_admedian['SINGLE_NUCLIOTIDE_POLYMORPHISM'] , color = 'green', linestyle = ':',\
     label='nucliotide' )
plt.plot([x**(1.3) for x in dfsc_admean['SDOH_COMMUNITY']],  color = 'y', linestyle = ':', \
    label = 'community' )
plt.plot([np.log(x) for x in dfsc_admean['SDOH_POVERTY']], color = 'r', linestyle = ':', \
    label = 'poverty' )
plt.plot([x**(1.32) for x in dfsc_admean['SDOH_ACTIVIY_LEVEL']], color = 'c', linestyle = ":",\
    label = 'SDOH_Activity')
plt.legend(loc = 'top right', prop = {'size' : 9}) 
#plt.title('Number of Hospitalization By Member Count')
plt.xlabel("Hospitalization Count")
plt.ylabel('Member Count')




#%% merge members to scores

#df_mem.
#df_score.shape
dffinal = df_mem.merge(df_score, left_on = 'id', right_on = 'id', how = 'inner')

#%% preprocessing and build model

dffinal2 = dffinal.merge(dfadmin_pivot, left_on = 'id', right_on = 'id', how = 'left')

#%%
#dffinal2.to_csv("processedfile.csv")

#%% encode & fill nan with zero

onehot = pd.get_dummies( dffinal2['gender'])
dffinal2 = dffinal2.drop('gender', axis =1)
dffinal2 = dffinal2.join(onehot)

#%% fill nan

dffinal2 = dffinal2.fillna(0)


#%% drop unused
#dffinal2.head()

dffinal2 = dffinal2.drop(['race', 'zip_code'], axis = 1)
#
dffinal2 = dffinal2.drop(['name', 'dob', 'id'], axis = 1)

#%% rename column 
dffinal2.rename(columns = {'admission_date': 'admission_count'}, inplace = True)


#%% bucket
def bucketfunction(x):
    if (x > 0) and (x <= 2):
        return 'onetwo'
    elif (x>2) and (x <= 5):
        return 'threefive'
    elif (x>5):
        return 'abovefive'
    else :
        return 'noadmit'   




#%% transform count to admission class

dffinal2['admission_class'] = [bucketfunction(x) for x in dffinal2['admission_count']]
dffinal2.head()

#%% drop admission count

dffinal2 = dffinal2.drop('admission_count', axis = 1)


#%% get data admitted class only

dffinal3 = dffinal2[dffinal2['admission_class']!= 'noadmit']

#%% preprocessing

from sklearn.model_selection import train_test_split
train,test = train_test_split(dffinal2,test_size = 0.3)

x_train = train.drop('admission_class', axis = 1)
y_train = train['admission_class']

x_test = test.drop('admission_class', axis = 1)
y_test = test['admission_class']


#%% load all classification 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV

#%% confusion matrix
from sklearn.metrics import confusion_matrix

#%% preprocessing for admit class only

from sklearn.model_selection import train_test_split
train,test = train_test_split(dffinal3,test_size = 0.3)

x_train1 = train.drop('admission_class', axis = 1)
y_train1 = train['admission_class']

x_test1 = test.drop('admission_class', axis = 1)
y_test1 = test['admission_class']

#%%
rfc = RandomForestClassifier()


#%% train

rfc.fit(x_train1, y_train1)


#%% test only admission

labels = np.unique(y_test1)
ypred_rc = rfc.predict(x_test1)
a = confusion_matrix(y_test1, ypred_rc, labels = labels)
admit_only = pd.DataFrame(a, index = labels, columns = labels)

#%% confusion matrix

admit_only.to_excel ('admitonly.xlsx')



#%% train model
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
#X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(x_train, y_train)

#%% make prediction

y_pred = clf.predict (x_test)


#%% prediction
y_pred1 = clf.predict(x_train)

#%% load confusion matrix

from sklearn.metrics import confusion_matrix
confusion_matrix (y_test, y_pred)

#%% confusion matrix on test

labels = np.unique(y_test)
a =  confusion_matrix(y_test, y_pred, labels=labels)

validation = pd.DataFrame(a, index=labels, columns=labels)


#%%  make confusion matrix
labels = np.unique(y_train)
b =  confusion_matrix(y_train, y_pred1, labels=labels)

testing = pd.DataFrame(b, index=labels, columns=labels)


#%% convert discharge date to date format

df_admin['discharge_date2'] = pd.to_datetime(df_admin['discharge_date'], infer_datetime_format=True)
df_admin.sort_values(['id', 'discharge_date2'], ascending = ('True', "True"), inplace = True)
df_admin.head()
#df_mem['dob1'] = pd.to_datetime(df_mem['dob'], format = '%m%d%y')




#%% print out feature importance
#Xsc = 

#train,test = train_test_split(dffinal2,test_size = 0.2)
#x_train = 
#y_train = 
importance = rfc.feature_importances_
columnname = list(x_train.columns)
indices = np.argsort(importance)[::-1]
print ('Feature ranking:')
for f in range(x_train.shape[1]):
    print("%d. feature %d (%f) --" % (f +1 , indices[f], importance[indices[f]]), columnname[f]) 


#%% feature with names feature
x_train.head()


feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(x_train.columns, rfc.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})

#%% sort value and resue

importances.sort_values('Gini-importance', ascending = (False), inplace = True)

importances.reset_index(inplace= True)


#%% save confusion matrix
importances.columns = ['Features', 'Gini-importance']
importances.to_excel("featureimportance.xlsx")
validation.to_excel ( 'allpop_confusionmatrix.xlsx')


#%%
from sklearn.cluster import KMeans
k = 5
kmeans = KMeans(n_clusters = k)
y_pred = kmeans.fit(dffinal4)

#%% plot the centroid

centroids = y_pred.cluster_centers_

plt.scatter(dffinal4['age'], dffinal4['ACG_CHF_IND_CD'], c= y_pred.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.xlabel('age')
plt.ylabel('ACG_CHF_IND_CD')







