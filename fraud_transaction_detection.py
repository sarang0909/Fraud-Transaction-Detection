
# coding: utf-8

# ## Goal

# E-commerce websites often transact huge amounts of money. And whenever a huge amount of money is moved, there is a high risk of users performing fraudulent activities, e.g. using stolen credit cards, doing money laundry, et. Machine Learning really excels at identifying fraudulent activities. Any website where you put your credit card information has a risk team in charge of avoiding frauds via machine learning. The goal of this challenge is to build a machine learning model that predicts the probability that the ﬁrst transaction of a new user is fraudulent.
# 

# ## Challenge Description

# Company XYZ is an e-commerce site that sells hand-made clothes. You have to build a model that predicts whether a user has a high probability of using the site to perform some illegal activity or not. This is a super common task for data scientists. You only have information about the user ﬁrst transaction on the site and based on that you have to make your classiﬁcation ("fraud/no fraud"). These are the tasks you are asked to do:<br> 1.For each user, determine her country based on the numeric IP address. <br>2.Build a model to predict whether an activity is fraudulent or not.<br>3. Explain how diﬀerent assumptions about the cost of false positives vs false negatives would impact the model.<br>4.Your boss is a bit worried about using a model she doesn't understand for something as important as fraud detection. How would you explain her how the model is making the predictions? Not from a mathematical perspective (she couldn't care less about that), but from a user perspective. What kinds of users are more likely to be classiﬁed as at risk? What are their characteristics?<br>5.Let's say you now have this model which can be used live to predict in real time if an activity is fraudulent or not. From a product perspective, how would you use it? That is,what kind of diﬀerent user experiences would you build based on the model output?
# 

# In[1]:

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:

#import dataset
fraud_data = pd.read_csv("Fraud_Data.csv")
ipAddress_to_country =pd.read_csv("IpAddress_to_Country.csv")


#  "Fraud_Data" - information about each user first transaction

# Columns:<br> user_id : Id of the user. Unique by user <br>signup_time : the time when the user created her account (GMT time)<br> purchase_time : the time when the user bought the item (GMT time)<br> purchase_value : the cost of the item purchased (USD)<br> device_id : the device id. You can assume that it is unique by device. I.e.,  transations with the same device ID means that the same physical device was used to buy <br>source : user marketing channel: ads, SEO, Direct (i.e. came to the site by directly typing the site address on the browser).<br> browser : the browser used by the user. <br>sex : user sex: Male/Female <br>age : user age <br>ip_address : user numeric ip address <br>class : this is what we are trying to predict: whether the activity was fraudulent (1) or not (0).
#  

#  "IpAddress_to_Country" - mapping each numeric ip address to its country. For each country, it gives a range. If the numeric ip address falls within the range, then the ip address belongs to the corresponding country.
# 

# Columns:<br> lower_bound_ip_address : the lower bound of the numeric ip address for that country <br>upper_bound_ip_address : the upper bound of the numeric ip address for that country <br>country : the corresponding country. If a user has an ip address whose value is within the upper and lower bound, then she is based in this country.
# 

# ### Data Exploration

# In[3]:

fraud_data.head()


# In[4]:

ipAddress_to_country.head()


# In[5]:

#Let's first add country column in fraud_data using ip_address


# In[6]:

fraud_data.info()


# In[7]:

ipAddress_to_country.info()


# In[8]:

#Check whether there is duplicate data for one user
len(fraud_data.user_id.unique()) == len(fraud_data)


# In[9]:

#Sample IP address
fraud_data.loc[0,"ip_address"]


# In[10]:

#First create country column based on Ip address
ls =list()
for i in np.arange(len(fraud_data)):
    ls = (fraud_data.ip_address[i]>=ipAddress_to_country.lower_bound_ip_address) & (fraud_data.ip_address[i]<=ipAddress_to_country.upper_bound_ip_address)
    #ls will be list of True and False,with only one True for above condition
    temp_country = ipAddress_to_country.loc[ls,"country"]
    #temp_country will be of type Series 
    if(len(temp_country)>=1):
        fraud_data.loc[i,"country"] = temp_country.unique()[0]
        '''Get unique value from Series and get first element'''


# In[11]:

fraud_data.head()


# In[260]:

#Let's create new fraud_data variable for data exploration
fraud_data_updated = fraud_data


# In[261]:

fraud_data_updated.info()


# In[262]:

'''
fraud_data_updated.class.value_counts()
File "<ipython-input-116-bac925af0542>", line 1
    fraud_data_updated.class.value_counts()
                           ^
SyntaxError: invalid syntax'''
#Learning:Don't use keywords as column names


# In[263]:

fraud_data_updated["class"].value_counts()


# In[264]:

'''Now lets remove un-necessary columns
1.user_id,device_id
2.ip_address as we already derived country from ip_address'''
fraud_data_updated = fraud_data_updated.drop(["user_id",'device_id','ip_address'],axis=1)
 


# In[265]:

fraud_data_updated.head()


# In[266]:

'''Now it is most likely that fraudster will spend very less time between signup and purchase
we can create new column time_spent=purchase_time-signup_time'''


# In[267]:

fraud_data_updated['time_spent'] = [(pd.to_datetime(fraud_data_updated.loc[i,'purchase_time']) - pd.to_datetime(
                                    fraud_data_updated.loc[i,'signup_time'])).total_seconds()/3600 
                                    for i in range(fraud_data_updated.shape[0])]


# In[268]:

fraud_data_updated.head()


# In[269]:

'''Let us normalize time_spent'''
min_time_spent = fraud_data_updated.time_spent.min()
max_time_spent = fraud_data_updated.time_spent.max()


# In[270]:

fraud_data_updated['time_spent'] = fraud_data_updated['time_spent'].apply(lambda x:(x-min_time_spent)/(max_time_spent-
                                                                                                       min_time_spent))


# In[271]:

fraud_data_updated.head()


# In[272]:

'''Remove signup time and purchase time columns'''
fraud_data_updated = fraud_data_updated.drop(['signup_time','purchase_time'],axis=1)


# In[273]:

fraud_data_updated.head()


# In[274]:

import pickle
pickle.dump((fraud_data_updated),open('preprocess.p', 'wb'))


# In[275]:

newset = pickle.load(open('preprocess.p', mode='rb'))


# In[276]:

'''Now let us analyse each column and its releation with class columns'''


# In[277]:

#Finding Correlation: Purchase value vs Class


# In[278]:

sns.boxplot(x=fraud_data_updated['class'],y=fraud_data_updated.purchase_value)


# In[279]:

sns.distplot(a=fraud_data_updated.purchase_value)


# In[280]:

new_set = fraud_data_updated[fraud_data_updated['class']==1]


# In[281]:

new_set.purchase_value.mean()


# In[282]:

new_set_two = fraud_data_updated[fraud_data_updated['class']==0]


# In[283]:

new_set_two.purchase_value.mean()


# In[284]:

'''As seen above mean of purchase value does not differ for different Class value ,
so it seems there is no relation between purchase value and Class,we will confirm it with Chi square test
For this we will convert Purchase value to categorical value'''


# In[285]:

purchase_value_cat = []
purchase_value_mean = fraud_data_updated.purchase_value.mean()
for i in range(len(fraud_data_updated.purchase_value)):
    if fraud_data_updated.loc[i,'purchase_value'] >= purchase_value_mean:
        purchase_value_cat.append('High')
    else:
        purchase_value_cat.append('Low')
fraud_data_updated['purchase_value_cat'] = purchase_value_cat


# In[286]:

crosstab = pd.crosstab(fraud_data_updated['purchase_value_cat'],fraud_data_updated['class'])


# In[287]:

from scipy import stats


# In[288]:

stats.chi2_contingency(crosstab)


# In[289]:

crosstab


# In[290]:

'''It is observed that although p value is not low,we will still keep purchase value column and maybe remove it if required after
validation testing'''


# In[291]:

fraud_data_updated = fraud_data_updated.drop('purchase_value_cat',axis=1)


# In[292]:

fraud_data_updated.head()


# In[293]:

#Finding Correlation: Source vs Class


# In[294]:

crosstab = pd.crosstab(fraud_data_updated['source'],fraud_data_updated['class'])


# In[295]:

crosstab


# In[296]:

stats.chi2_contingency(crosstab)


# In[297]:

'''so low p value(second value 7.98) clearly indicates strong correlation between Source and class'''


# In[298]:

fraud_data_updated.head()


# In[299]:

#Finding Correlation: browser vs Class


# In[300]:

crosstab = pd.crosstab(fraud_data_updated['browser'],fraud_data_updated['class'])


# In[301]:

crosstab  


# In[302]:

stats.chi2_contingency(crosstab)


# In[303]:

'''so low p value(second value) clearly indicates strong correlation between browser and class'''


# In[304]:

fraud_data_updated.head()


# In[305]:

#Finding Correlation: sex vs Class


# In[306]:

crosstab = pd.crosstab(fraud_data_updated['sex'],fraud_data_updated['class'])


# In[307]:

crosstab


# In[308]:

stats.chi2_contingency(crosstab)


# In[309]:

'''so low p value(second value) clearly indicates strong correlation between sex and class'''


# In[310]:

fraud_data_updated.head()


# In[311]:

#Finding Correlation: country vs Class


# In[312]:

fraud_data_updated.country.isna().sum()


# In[313]:

'''Earlier we were unable to identify country for so many rows from IP address,but we can not simply ignore 
these NA values in country.It might me the case that user is accessing from unknown/fraudent IP address,So we will
create new Country/Category for these users as UNKNOWN'''


# In[314]:

fraud_data_updated['country'] = fraud_data_updated.country.fillna('Unknown')


# In[315]:

fraud_data_updated.head()


# In[316]:

crosstab = pd.crosstab(fraud_data_updated['country'],fraud_data_updated['class'])


# In[317]:

crosstab


# In[318]:

stats.chi2_contingency(crosstab)


# In[319]:

'''so low p value(second value) clearly indicates strong correlation between sex and class'''


# In[320]:

fraud_data_updated.head()


# In[321]:

#Encode categorical features


# In[322]:

source_dummy = pd.get_dummies(fraud_data_updated['source'])
fraud_data_updated = pd.concat([fraud_data_updated,source_dummy],axis=1)


# In[323]:

browser_dummy = pd.get_dummies(fraud_data_updated['browser'])
fraud_data_updated = pd.concat([fraud_data_updated,browser_dummy],axis=1)


# In[324]:

sex_dummy = pd.get_dummies(fraud_data_updated['sex'])
fraud_data_updated = pd.concat([fraud_data_updated,sex_dummy],axis=1)


# In[325]:

country_dummy = pd.get_dummies(fraud_data_updated['country'])
fraud_data_updated = pd.concat([fraud_data_updated,country_dummy],axis=1)


# In[326]:

fraud_data_updated = fraud_data_updated.drop(['sex','browser','source','country'],axis=1)


# In[327]:

fraud_data_updated.head()


# In[328]:

fraud_data_updated.info()


# In[329]:

fraud_data_updated.shape


# In[330]:

pickle.dump((fraud_data_updated),open('preprocess2.p', 'wb'))
#fraud_data_updated = pickle.load(open('preprocess_second.p', mode='rb'))


# In[331]:

feature_columns = fraud_data_updated.columns.values


# In[333]:

feature_columns


# ### Create features and target

# In[335]:

target = np.array(fraud_data_updated['class'])
features = np.array(fraud_data_updated.drop(['class'],axis=1))


# In[336]:

target.shape


# In[337]:

#target = target.reshape(-1,1)


# In[338]:

target.shape


# In[339]:

features.shape


# In[2]:

import pickle
#pickle.dump((features,target),open('preprocess_data.p', 'wb'))
features,target= pickle.load(open('preprocess_data.p', mode='rb'))


# ### Create training and test set

# In[4]:

from sklearn.model_selection import StratifiedShuffleSplit
splitObject = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index, test_index in splitObject.split(features, target):
    X_train, X_test = features[train_index], features[test_index] 
    y_train, y_test = target[train_index], target[test_index]


# ### Create Models

# We will try following models and then choose the best model by comparing cross validation accuracy,then fine tune the model<br>
# 1.Logistic regression <br>
# 2.KNN<br>
# 3.SGD Classifier<br>
# 4.Random Forest
# 
# 
# 

# In[7]:

#1 Logistic Regression


# In[8]:

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,recall_score


# In[9]:

lg_model = LogisticRegression()
lg_model.fit(X_train,y_train)
lg_y_pred = lg_model.predict(X_train)


# In[10]:

confusion_matrix(y_train,lg_y_pred)


# Recall is more important than precision,as we want to have very less false negative

# In[12]:

print(cross_val_score(lg_model,X_train,y_train,cv=3,scoring='recall'))


# In[13]:

print(classification_report(y_train, lg_y_pred))


# In[14]:

lg_model


# In[15]:

# 2 KNN


# In[16]:

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()
knn_model.fit(X_train,y_train)
knn_y_pred = knn_model.predict(X_train)


# In[17]:

confusion_matrix(y_train,knn_y_pred)


# In[18]:

print(cross_val_score(knn_model,X_train,y_train,cv=3,scoring='recall'))


# In[21]:

# 3 SGD Classifier


# In[30]:

from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier()
sgd_model.fit(X_train,y_train)
sgm_y_pred = sgd_model.predict(X_train)


# In[31]:

confusion_matrix(y_train,sgm_y_pred)


# In[32]:

print(cross_val_score(sgd_model,X_train,y_train,cv=3,scoring='recall'))


# In[33]:

# 4 Random Forest


# In[34]:

from sklearn.ensemble import RandomForestClassifier


# In[35]:

forest_model = RandomForestClassifier()
forest_model.fit(X_train,y_train)
forest_y_pred = forest_model.predict(X_train)


# In[36]:

confusion_matrix(y_train,forest_y_pred)


# In[37]:

print(cross_val_score(forest_model,X_train,y_train,cv=3,scoring='recall'))


# It can be seen as RandomForest works best,we can optimize it further by finding tuning parameteres

# ### Model Tuning

# In[38]:

forest_model = RandomForestClassifier()
forest_model.fit(X_train,y_train)
print(cross_val_score(forest_model,X_train,y_train,cv=3,scoring='recall'))


# In[39]:

forest_model


# Let's find tuning parameter through grid search

# In[42]:

from sklearn.model_selection import GridSearchCV


# In[44]:

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]
grid_search = GridSearchCV(forest_model,param_grid=param_grid,cv=5)


# In[45]:

grid_search.fit(X_train,y_train)


# In[48]:

grid_search.best_estimator_


# In[46]:

tuned_forest_model = grid_search.best_estimator_


# In[47]:

print(cross_val_score(tuned_forest_model,X_train,y_train,cv=3,scoring='recall'))


# In[50]:

y_pred = tuned_forest_model.predict(X_test)
confusion_matrix(y_test,y_pred)


# In[51]:

from sklearn.metrics import recall_score
recall_score(y_test, y_pred, average='binary')


# Now model is tuned but still we can try ensemble learning model

# ### Ensemble Techniques
# 1.Voting<br>
# 2.Bagging<br>
# 3.Boosting

# In[59]:

# 1.Voting 
'''Let us use classifiers created above'''
from sklearn.ensemble import VotingClassifier
models = [lg_model,knn_model,forest_model]
voting = VotingClassifier(estimators=[('lg', lg_model), ('knn', knn_model), ('forest',forest_model)],voting='hard')
voting.fit(X_train,y_train)
y_pred = voting.predict(X_test)
recall_score(y_test, y_pred, average='binary')
 


# In[61]:

#2.Bagging
'''We have already used one bagging algorithm i.e RandomForest,so let us use different bagging algo'''
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
bagging = BaggingClassifier(base_estimator=dt)
bagging.fit(X_train,y_train)
y_pred = bagging.predict(X_test)
recall_score(y_test, y_pred, average='binary')


# In[62]:

#Boosting
'''AdaBoost'''
from sklearn.ensemble import AdaBoostClassifier
ada_boost = AdaBoostClassifier()
ada_boost.fit(X_train,y_train)
y_pred = ada_boost.predict(X_test)
recall_score(y_test, y_pred, average='binary')


# In[63]:

'''Gradient Boosting machine'''
from sklearn.ensemble import GradientBoostingClassifier
grad_boost = GradientBoostingClassifier()
grad_boost.fit(X_train,y_train)
y_pred = grad_boost.predict(X_test)
recall_score(y_test, y_pred, average='binary')


# In[65]:

'''xgboost,lightboost not part of sklearn'''
#import xgboost as xgb
#xg_boost=xgb.XGBClassifier(learning_rate=0.01)


# ### Precision-Recall Curve

# In[ ]:




# In[73]:

from sklearn.metrics import precision_recall_curve
probs = grad_boost.predict_proba(X_test)
preds = probs[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, preds)

plt.title('Precision Recall Curve')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
plt.plot(recall, precision, marker='.')


# ### ROC curve

# In[68]:

from sklearn.metrics import roc_curve,auc
probs = grad_boost.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:



