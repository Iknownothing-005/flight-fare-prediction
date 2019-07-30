#!/usr/bin/env python
# coding: utf-8

# In[1041]:


import os
os.chdir("E:\Hackathon\Flight_Ticket_Participant_Datasets")


# In[1042]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[1043]:


data = pd.read_excel("Data_train.xlsx")
data_test = pd.read_excel("Test_set.xlsx")


# In[1044]:


X = data.append(data_test)
X.reset_index(inplace = True)


# In[1045]:


X.info()


# In[1046]:


missing_values  = pd.DataFrame(X.isnull().sum())
missing_values


# In[1047]:


X.dropna(subset = ["Total_Stops"],inplace = True)
missing_values  = pd.DataFrame(X.isnull().sum())
missing_values


# In[1048]:


X.info()


# In[1049]:


target = X["Price"]


# In[1050]:


X.drop(["Price"],axis = 1, inplace = True)
X.drop(["index"],axis = 1, inplace = True)
X.shape


# In[1051]:


X["Date_of_Journey"] = pd.to_datetime(X["Date_of_Journey"])
X["Dep_Time"] = pd.to_datetime(X["Dep_Time"])
X["Arrival_Time"] = pd.to_datetime(X["Arrival_Time"])


# In[1052]:


X.info()


# In[1053]:


X["Duration"] = X["Duration"].map(lambda x: x.replace("h",""))
X["Duration"] = X["Duration"].map(lambda x: x.replace("m",""))
X["Duration"] = X["Duration"].map(lambda x: x[0:2])


# In[1054]:


X["Stop_1"] = X["Route"].str.split("→").str[1]
X["Stop_2"] = X["Route"].str.split("→").str[2]
X["Stop_1"] = np.where(X["Total_Stops"] == "non-stop","none",X["Stop_1"])
X["Stop_2"] = np.where(X["Total_Stops"] == "non-stop","none",X["Stop_2"])
X["Stop_2"] = np.where(X["Total_Stops"] == "1 stop","none",X["Stop_2"])
X.head()


# In[1055]:


X["Day_of_Booking"] = X["Date_of_Journey"].dt.weekday_name
X["month_of_Booking"] = X["Date_of_Journey"].dt.month_name().str.slice(stop=3)
X["Dep_Time_Hour"] = X["Dep_Time"].dt.hour
X["Arrival_Time_hour"] = X["Arrival_Time"].dt.hour

X.drop(["Date_of_Journey"],axis = 1,inplace = True)
X.drop(["Route"],axis = 1,inplace = True)
X.drop(["Dep_Time"],axis = 1,inplace = True)
X.drop(["Arrival_Time"],axis = 1,inplace = True)
X.head()


# In[1056]:


X.info()


# In[1057]:


#creating dummies for all categorical classes 
def get_dummies(dataset,columns):
    for j in columns:
        dummies = pd.get_dummies(dataset[j], prefix = j)
        dataset = pd.concat([dataset, dummies], axis = 1)
        dataset.drop(j,axis =1, inplace = True)
    return dataset


# In[1058]:


cat_col = ["Airline","Source","Destination","Total_Stops","Additional_Info","Day_of_Booking","month_of_Booking","Stop_1","Stop_2"]
X = get_dummies(X,cat_col)


# In[1059]:


X.info()


# In[1060]:


X.drop(["Airline_Trujet"],axis = 1,inplace = True)
X.drop(["Source_Chennai"],axis = 1,inplace = True)
X.drop(["Destination_Kolkata"],axis = 1,inplace = True)
X.drop(["Total_Stops_4 stops"],axis = 1,inplace = True)
X.drop(["Additional_Info_1 Short layover"],axis = 1,inplace = True)
X.drop(["Day_of_Booking_Sunday"],axis = 1,inplace = True)
X.drop(["month_of_Booking_Apr"],axis = 1,inplace = True)


# In[1061]:


X["Duration"] = X["Duration"].apply(pd.to_numeric)
#coverting segregrating hours to group and creating dummies for data 
X["Arrival_Morning"] = X["Arrival_Time_hour"].map(lambda x: 1 if 12 > x >=6 else 0 )
X["Arrival_afternoon"] = X["Arrival_Time_hour"].map(lambda x: 1 if 12 <= x < 17 else 0 )
X["Arrival_evening"] = X["Arrival_Time_hour"].map(lambda x: 1 if 17 <= x < 20 else 0 )
X["Arrival_night"] = X["Arrival_Time_hour"].map(lambda x: 1 if 20 <= x < 23 else 0 )
X["Arrival_late_night"] = X["Arrival_Time_hour"].map(lambda x: 1 if 0 <= x < 3 else 0 )
X["Arrival_early_morning"] = X["Arrival_Time_hour"].map(lambda x: 1 if 3 <= x < 6 else 0 )
#coverting segregrating hours to group and creating dummies for data 
X["Dept_Morning"] = X["Dep_Time_Hour"].map(lambda x: 1 if 12 > x >=6 else 0 )
X["Dept_afternoon"] = X["Dep_Time_Hour"].map(lambda x: 1 if 12 <= x < 17 else 0 )
X["Dept_evening"] = X["Dep_Time_Hour"].map(lambda x: 1 if 17 <= x < 20 else 0 )
X["Dept_night"] = X["Dep_Time_Hour"].map(lambda x: 1 if 20 <= x < 23 else 0 )
X["Dept_late_night"] = X["Dep_Time_Hour"].map(lambda x: 1 if 0 <= x < 3 else 0 )
X["Dept_early_morning"] = X["Dep_Time_Hour"].map(lambda x: 1 if 3 <= x < 6 else 0 )
X.drop(["Arrival_Time_hour"],axis = 1,inplace = True)
X.drop(["Dep_Time_Hour"],axis = 1,inplace = True)
X.drop(["Stop_1_ JLR "],axis = 1,inplace = True)
X.drop(["Stop_2_ CCU "],axis = 1,inplace = True)
X.info()


# In[1062]:


X.drop(["Arrival_early_morning"],axis = 1,inplace = True)
X.drop(["Dept_late_night"],axis = 1,inplace = True)


# In[1063]:


X.shape


# In[1064]:


# Usng Backward Elimination method with multiple linear 
from scipy.stats import chi2_contingency
cat_col = list(X.iloc[:,1:124].columns)
pmax = 1
while (len(cat_col)>0):
    v= []
    for i in cat_col:
        chi2,p,dof,ex  = chi2_contingency(pd.crosstab(data["Price"],X[i]))
        v.append(p)
    v = pd.Series(v)    
    x = pd.Series(v.values,index = cat_col)     
    pmax = max(x)
    feature_with_p_max = x.idxmax()
    if(pmax>0.05):
        cat_col.remove(feature_with_p_max)
    else:
        break

selected_features_BE1 = cat_col
print(selected_features_BE1)
print(x)


# In[1065]:


T = X[selected_features_BE1]
train = T.iloc[:10682,:]


# In[1071]:


type(target)
target.dropna(inplace = True)


# In[1074]:


import statsmodels.formula.api as sm
L = X.iloc[:10682,:]
cols = list(L.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_opt = L[cols]
    X_opt = np.append(arr = np.ones((10682,1)).astype(int), values = X_opt,axis=1)
    Reg_OLS = sm.OLS(target,X_opt).fit()
    p = pd.Series(Reg_OLS.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break

selected_features_BE = cols
print(selected_features_BE)
print(p)


# In[1078]:


x_ols = L.loc[:,selected_features_BE]
x_ols.shape


# In[1079]:


from sklearn.preprocessing import normalize
x = normalize(train, axis=0, norm='max')
x_ols = normalize(x_ols, axis = 0, norm = 'max')


# In[1080]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,target,test_size=0.35,random_state = 0)
x_train_ols,x_test_ols,y_train,y_test = train_test_split(x_ols,target,test_size=0.35,random_state = 0)


# In[1081]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 50)


# In[1082]:


x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_


# In[1083]:


from sklearn.linear_model import LinearRegression
classifier1 = sm.OLS(endog = y_train, exog = x_train).fit()
y_pred1 = classifier1.predict(x_test)


# In[1084]:


from sklearn.metrics import mean_squared_error as mse
rmse = np.sqrt(mse(y_pred1,y_test))
print(rmse)


# In[1085]:


classifier2 = sm.OLS(endog = y_train, exog = x_train_pca).fit()
y_pred2 = classifier2.predict(x_test_pca)


# In[1086]:


rmse = np.sqrt(mse(y_pred2,y_test))
print(rmse)


# In[1087]:


classifier3 = sm.OLS(endog = y_train, exog = x_train_ols).fit()
y_pred3 = classifier3.predict(x_test_ols)


# In[1088]:


rmse = np.sqrt(mse(y_pred3,y_test))
print(rmse)


# In[1089]:


from sklearn.svm import SVR
svr_reg1  = SVR()
svr_reg1.fit(x_train,np.ravel(y_train))
ypred_reg1 = svr_reg1 .predict(x_test)


# In[1090]:


rmse = np.sqrt(mse(ypred_reg1,y_test))
print(rmse)


# In[1091]:


svr_reg2 = SVR()
svr_reg2.fit(x_train_ols,np.ravel(y_train))
ypred_reg2 = svr_reg2.predict(x_test_ols)


# In[1092]:


rmse = np.sqrt(mse(ypred_reg2,y_test))
print(rmse)


# In[1093]:


svr_reg3 = SVR()
svr_reg3.fit(x_train_pca,np.ravel(y_train))
ypred_reg3 = svr_reg3.predict(x_test_pca)


# In[1094]:


rmse = np.sqrt(mse(ypred_reg3,y_test))
print(rmse)


# In[1095]:


from sklearn.tree import DecisionTreeRegressor 
DT_reg1 = DecisionTreeRegressor(random_state = 0)
DT_reg1.fit(x_train,y_train)
ypred_dt1 = DT_reg1.predict(x_test)


# In[1096]:


rmse = np.sqrt(mse(ypred_dt1,y_test))
print(rmse)


# In[1097]:


DT_reg2 = DecisionTreeRegressor(random_state = 0)
DT_reg2.fit(x_train_ols,y_train)
ypred_dt2 = DT_reg2.predict(x_test_ols)


# In[1098]:


rmse = np.sqrt(mse(ypred_dt2,y_test))
print(rmse)


# In[1099]:


DT_reg3 = DecisionTreeRegressor(random_state = 0)
DT_reg3.fit(x_train_pca,y_train)
ypred_dt3 = DT_reg3.predict(x_test_pca)


# In[1100]:


rmse = np.sqrt(mse(ypred_dt3,y_test))
print(rmse)


# In[1101]:


from sklearn.ensemble import RandomForestRegressor
RF_reg1 = RandomForestRegressor(random_state=0)
RF_reg1.fit(x_train,np.ravel(y_train))
ypred_RF1 = RF_reg1.predict(x_test)


# In[1102]:


rmse = np.sqrt(mse(ypred_RF1,y_test))
print(rmse)
from sklearn.metrics import mean_squared_log_error
rmsle = np.sqrt(mean_squared_log_error(ypred_RF1,y_test))
print(rmsle)


# In[1103]:


RF_reg2 = RandomForestRegressor(random_state=0)
RF_reg2.fit(x_train_ols,np.ravel(y_train))
ypred_RF2 = RF_reg2.predict(x_test_ols)


# In[1104]:


rmse = np.sqrt(mse(ypred_RF2,y_test))
print(rmse)
rmsle = np.sqrt(mean_squared_log_error(ypred_RF2,y_test))
print(rmsle)


# In[1105]:


RF_reg3 = RandomForestRegressor(random_state=0)
RF_reg3.fit(x_train_pca,np.ravel(y_train))
ypred_RF3 = RF_reg3.predict(x_test_pca)


# In[1106]:


rmse = np.sqrt(mse(ypred_RF3,y_test))
print(rmse)


# In[1107]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
RF_reg4 = RandomForestRegressor()
score = cross_val_score(RF_reg4,x_train_ols,y_train,scoring = "neg_mean_squared_error", cv = 10, n_jobs = -1).mean()
print((-score))


# In[1108]:


rmse = np.sqrt(-(score))
rmse


# In[1109]:


parameters = [{"n_estimators":[10,13,15],"max_depth":[10,15,18],"min_samples_leaf":[2,5,10]}]


# In[1110]:


grid_search = GridSearchCV(estimator = RF_reg4,
                          param_grid = parameters,
                          scoring =  "neg_mean_squared_log_error",
                          cv = 10,
                          n_jobs = -1)


# In[1111]:


grid_search_rf = grid_search.fit(x_train,np.ravel(y_train))


# In[1113]:


best_parameters = grid_search_rf.best_params_


# In[1114]:


best_parameters


# In[1115]:


best_accuracy = grid_search_rf.best_score_
rmsle = (np.sqrt(-(best_accuracy)))
rmsle


# In[1116]:


from xgboost import  XGBRegressor


# In[1117]:


xgb_model =  XGBRegressor(learning_rate = 0.1, n_estimators = 200, max_depth = 10, min_child_weight = 6,
               gamma = 0,subsample = 0.7,colsample_bytree = 0.8, objective = "reg:squarederror",
               nthread = 4, scale_pos_weigth = 1,seed = 27, reg_alpha = 0.00006)


# In[1118]:


xgb_model.fit(x_train,np.ravel(y_train))
y_pred_xgb = xgb_model.predict(x_test)


# In[1119]:


rmse = np.sqrt(mse(y_pred_xgb,y_test))
print(rmse)


# In[1120]:


parameters = {'colsample_bytree': [0.8],
              'learning_rate': [0.1],
              'max_depth': [7,8,10],
              'min_child_weight': [4,6,8],
              'n_estimators': [200],
              'nthread': [4],
              'objective': ["reg:squarederror"],
              'silent': [1],
              'subsample':[ 0.7],
              'gamma':[0],
              'reg_alpha':[0.00006],
              'scale_pos_weight':[1]}


# In[1121]:


xgb_grid = XGBRegressor() 
grid_search = GridSearchCV(estimator = xgb_grid,
                          param_grid = parameters,
                          scoring =  "neg_mean_squared_log_error",
                          cv = 10,
                          n_jobs = -1)


# In[1122]:


grid_xgb = grid_search.fit(x_train,np.ravel(y_train))


# In[1123]:


best_parameters = grid_xgb.best_params_
best_parameters


# In[1124]:


best_accuracy = grid_xgb.best_score_
best_accuracy = (np.sqrt(-(best_accuracy)))
best_accuracy


# In[1125]:


grid_final = XGBRegressor(**grid_xgb.best_params_) 
grid_final.fit(train,np.ravel(target))


# In[1126]:


test = X.iloc[10682:13353,:]
test = test[selected_features_BE1]


# In[1127]:


test.shape


# In[1128]:


prediction = grid_final.predict(test)


# In[1129]:


prediction


# In[1130]:


prediction_df = pd.DataFrame(prediction)


# In[1131]:


prediction_df.to_excel("Prediction_new.xlsx", index = False)


# In[1]:


get_ipython().system('jupyter nbconvert --to script config_template.ipynb')


# In[ ]:




