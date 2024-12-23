#!/usr/bin/env python
# coding: utf-8

# ## Loan Status Prediction Using ML

# In[2]:


import pandas as pd


# In[3]:


data = pd.read_csv("loan_prediction.csv")


# In[4]:


# COLUMN DETAILS

# Loan_ID : Unique Loan ID

# Gender : Male/ Female

# Married : Applicant married (Y/N)

# Dependents : Number of dependents

# Education : Applicant Education (Graduate/ Under Graduate)

# Self_Employed : Self employed (Y/N)

# ApplicantIncome : Applicant income

# CoapplicantIncome : Coapplicant income

# LoanAmount : Loan amount in thousands of dollars

# Loan_Amount_Term : Term of loan in months

# Credit_History : Credit history meets guidelines yes or no

# Property_Area : Urban/ Semi Urban/ Rural

# Loan_Status : Loan approved (Y/N) this is the target variable


# ### 1. Display Top 5 Rows of The Dataset

# In[6]:


data.head()


# ### 2. Check Last 5 Rows of The Dataset

# In[8]:


data.tail()


# ### 3. Find Shape of Our Dataset (Number of Rows And Number of Columns)

# In[10]:


data.shape


# In[11]:


print("Number of Rows: ",data.shape[0])
print("Number of Columns: ",data.shape[1])


# ### 4. Get Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column And Memory Requirement

# In[13]:


data.info()


# ### 5. Check Null Values In The Dataset

# In[15]:


data.isnull().sum()


# In[16]:


# Missing Percentage

data.isnull().sum()*100 / len(data)


# ### 6. Handling The missing Values

# In[18]:


# dropping Loan_ID column entirely

data = data.drop('Loan_ID', axis=1)


# In[19]:


data.head(1)


# In[20]:


# making a list of columns with missing percentage < 5%

columns = ['Gender','Dependents','LoanAmount','Loan_Amount_Term']


# In[21]:


# dropping rows and columns with missing percentage less than 5%

data = data.dropna(subset=columns) # as default axis=0 (meaning drop rows which contain missing values)


# In[22]:


# checking missing percentage again

data.isnull().sum()*100 / len(data)


# - All columns, except **'Self_Employed'** and **'Credit_History'** are handled and these column's missing percentage is more than 5%, so we can't delete row them, we've to fill the missing values with appropriate values.
# 

# In[24]:


data['Self_Employed'].unique()


# In[25]:


data['Credit_History'].unique()


# In[26]:


data['Self_Employed'].mode()[0]


# In[27]:


data['Credit_History'].mode()[0]


# In[28]:


data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])


# In[29]:


data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])


# In[30]:


data.isnull().sum()*100 / len(data)


# - **All missing values are handled.**

# ### 7. Handling Categorical Columns

# In[33]:


data.sample(5)


# In[34]:


# replace 3+ with 3

data['Dependents'] = data['Dependents'].replace(to_replace="3+",value='3')


# In[35]:


data['Dependents'].unique()


# In[36]:


data['Loan_Status'].unique()


# #### Encoding
# 
# #As machines only understand 0's and 1's. We've to convert our categorical columns to 0's and 1's.

# In[38]:


data['Gender'] = data['Gender'].map({'Male':1,'Female':0}).astype('int')
data['Married'] = data['Married'].map({'Yes':1,'No':0}).astype('int')
data['Education'] = data['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
data['Self_Employed'] = data['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
data['Property_Area'] = data['Property_Area'].map({'Rural':0,'Semiurban':2,'Urban':1}).astype('int')
data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0}).astype('int')


# In[39]:


data.head()


# ### 8. Store Feature Matrix In X And Response (Target) In Vector y

# In[41]:


X = data.drop('Loan_Status', axis=1)


# In[42]:


y = data['Loan_Status']


# ### 9. Feature Scaling

# In[44]:


X.head()


# - *Gender, Married, Dependents, Education, Self_Employed, Credit_History, Property_Area* values are in the same range.
# - Scaling the rest. If these are not scale, then Features with higher value range starts dominating in calculating distances b/w features.
# 
# - **Distance based Algorithms:**<br>
#  1. K-nearest Neighbour <br>
#  2. Neural Networking. <br>
#  3. Support vector machine. <br>
#  4. Linear Regression. <br>
#  5. Logistic Regression. <br>
# 
# - ML algorithm which don't need feature scaling are **Non-linear algorithms**. like *Decision Tree, Random Forest, Gradient Bost, Naive Bayes*, etc.
# - Any algorithm, which is not distance based is not affected by feature scaling. 

# In[46]:


# making a list of columns that we need to scale

cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']


# In[47]:


from sklearn.preprocessing import StandardScaler

st = StandardScaler()
X[cols] = st.fit_transform(X[cols])


# In[48]:


X


# ### 10. Splitting The Dataset Into The Training Set And Test Set & Applying K-Fold Cross Validation 

# In[50]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np


# In[51]:


model_df = {}

def model_val(model,X,y):
    # spliting dataset for training and testing
    X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                   test_size=0.20,
                                                   random_state=42)
    
    # training the model
    model.fit(X_train, y_train)
    
    # asking model for prediction
    y_pred = model.predict(X_test)
    
    # checking model's prediction accuracy
    print(f"{model} accuracy is {accuracy_score(y_test,y_pred)}")
    
    # to find the best model we use cross-validation, thru this we can compare different algorithms
    # In this we use whole dataset to for testing not just 20%, but one at a time and summarize 
    # the result at the end.
    
    # 5-fold cross-validation (but 10-fold cross-validation is common in practise)
    score = cross_val_score(model,X,y,cv=5)  # it will divides the dataset into 5 parts and during each iteration 
                                             # uses (4,1) combination for training and testing 
    
    print(f"{model} Avg cross val score is {np.mean(score)}")
    model_df[model] = round(np.mean(score)*100,2)
    


# ### 11. Logistic Regression

# In[53]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# passing this model object of LogisticRegression Class in the function we've created
model_val(model,X,y)


# In[54]:


model_df


# ### 12. SVC (Support Vector Classifier)

# In[56]:


from sklearn import svm

model = svm.SVC()
model_val(model,X,y)


# In[57]:


model_df


# ### 13. Decision Tree Classifier

# In[59]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model_val(model,X,y)


# In[60]:


model_df


# ### 14. Random Forest Classifier

# In[62]:


from sklearn.ensemble import RandomForestClassifier

model =RandomForestClassifier()
model_val(model,X,y)


# In[63]:


model_df


# ### 15. Gradient Boosting Classifier

# In[65]:


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model_val(model,X,y)


# In[66]:


model_df


# - Clearly, **LogisticRegression is the best model with accuracy 80.48.**
# - ***But, we've trained our model with default parameters of Logistic Regression. Same with other algorithms as well.***

# ### 16. HyperParameter Tuning

# In ML, there are two types of parameters:
#    1. Model parameters
#    2. Hyper parameters
#     
# **Model Parameters:** are parameters that model will learn during training phase. <br>
#   For Example, y = mx + c. For Linear Regression, model will learn 'm' and 'c' during training phase.
# so, **m** and **c** are called as **model parameters.**
# 
# **Hyper Parameters:** This are adjustable parameters that must by tuned in order to obtain a model with optimal performance.<br>
#                   ML models can have many hyper parameters and finding best combination of parameters can be treated as **"Search Problem"**.
#         
# There are two best strategies for hyper parameter tuning.
# 1. **Grid Search CV** - go thru all the intermediate combination of parameters which make it computationally very expensive.
# 2. **Randomized Search CV** - it solve the drawbacks of GridSearchCV, as it goes thru only fixed no. of hyper parameter settings. it moves within a grid in random fashion to find best set of hyper parameter.
# 
# We'll we using Randomized Search CV

# In[70]:


from sklearn.model_selection import RandomizedSearchCV


# ### Logistic Regression

# In[72]:


# Let's tune hyper parameters of LogisticRegression (we've choosen 'C' and 'solver' parameter for tuning)

log_reg_grid = {"C": np.logspace(-4,4,20),
                "solver": ['liblinear']}


# In[73]:


# In RandomizedSearchCV we've to pass estimator, which is nothing but Algo class, It will return
# a model with it's Hyper Parameter already set and we've to train that model, with our dataset

rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                   param_distributions = log_reg_grid,
                   n_iter=20, cv=5, verbose=True)


# In[74]:


# Let's train our model with these set hyper parameters for optimized results.

rs_log_reg.fit(X,y)


# In[75]:


rs_log_reg.best_score_


# In[76]:


rs_log_reg.best_params_


# ### SVC (Support Vector Classifier)

# In[78]:


svc_grid = {'C':[0.25,0.50,0.75,1],
            "kernel":["linear"]}


# In[79]:


rs_svc=RandomizedSearchCV(svm.SVC(),
                  param_distributions = svc_grid,
                  cv=5,
                  n_iter=20,
                  verbose=True)


# In[80]:


rs_svc.fit(X,y)


# In[81]:


rs_svc.best_score_


# - **Earlier it was 79.39. So there is some improvement.**

# In[83]:


rs_svc.best_params_


# ###  Random Forest Classifier

# In[85]:


rf_grid = {'n_estimators':np.arange(10,1000,10),
           'max_features':['log2','sqrt'],
           'max_depth':[None,3,5,10,20,30],
           'min_samples_split':[2,5,20,50,100],
           'min_samples_leaf':[1,2,5,10]
          }


# In[86]:


rs_rf = RandomizedSearchCV(RandomForestClassifier())


# In[87]:


rs_rf.fit(X,y)


# In[88]:


rs_rf.best_score_


# In[89]:


rs_rf.best_params_


# In[90]:




# > Selecting **Random Forest Classifier** for Production with **80.66** accuracy. Earlier it was Logistic Regression with 80.48.

# ### 17. Save The Model

# **Before we save our best model RandomForestClassifier for Production, we've to train our Model on the entire dataset with best parameters that we've just found.**

# In[ ]:


X = data.drop('Loan_Status',axis=1)
y = data['Loan_Status']


# In[ ]:


rf = RandomForestClassifier(n_estimators = 270,
                            min_samples_split = 5,
                            min_samples_leaf = 5,
                            max_features = 'sqrt',
                            max_depth = 5)


# In[ ]:


rf.fit(X,y)


# #### Saving our model, so that we've don't have to train it again.

# In[ ]:


import joblib


# In[ ]:


# saving our model by passing an instance of our model and giving it a name.

joblib.dump(rf,'loan_status_prediction_model')


# In[ ]:


# In Future, we can perform predictin using this saved model, as shown below


from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS library
import joblib  # or pickle
import numpy as np

# Load your trained model
model = joblib.load('loan_prediction_model.pkl')

# Create Flask app
app = Flask(__name__)
CORS(app)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        result = {
            'prediction': int(prediction[0])  # Convert NumPy int64 to Python int
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=True)




