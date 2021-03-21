#Installation of required libraries
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors

from warnings import filterwarnings
filterwarnings('ignore')

#Reading the dataset
df = pd.read_csv("Hitters.csv")

# The variables related to their careers were divided into career years and new values were created in the data set by obtaining average values.
df["OrtCAtBat"] = df["CAtBat"] / df["Years"]
df["OrtCHits"] = df["CHits"] / df["Years"]
df["OrtCHmRun"] = df["CHmRun"] / df["Years"]
df["OrtCruns"] = df["CRuns"] / df["Years"]
df["OrtCRBI"] = df["CRBI"] / df["Years"]
df["OrtCWalks"] = cwalks = df["CWalks"] / df["Years"]

# Based on some trials and correlation results, we subtract variables that do not contribute to the model from our data set.
df = df.drop(['AtBat','Hits','HmRun','Runs','RBI','Walks','Assists','Errors',"PutOuts",'League','NewLeague', 'Division'], axis=1)

#We fill in the missing observations with the KNN method.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors = 4)
df_filled = imputer.fit_transform(df)

df = pd.DataFrame(df_filled,columns = df.columns)

Q1 = df.Salary.quantile(0.25)
Q3 = df.Salary.quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df.loc[df["Salary"] > upper,"Salary"] = upper

from sklearn.neighbors import LocalOutlierFactor
lof =LocalOutlierFactor(n_neighbors= 10)
lof.fit_predict(df)
df_scores = lof.negative_outlier_factor_
threshold = np.sort(df_scores)[7]
outlier = df_scores > threshold
df = df[outlier]

y = df["Salary"]
X = df.drop("Salary",axis=1)

cols = X.columns

import statsmodels.api as sm
from sklearn.feature_selection import RFE
#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
X = df[selected_features_BE]
X = df[["CRuns","OrtCWalks","CWalks"]]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
gbm_cv_model_best_params_ ={'learning_rate': 0.01,'loss': 'lad','max_depth': 5,'n_estimators': 500,'subsample': 0.5}
gbm_tuned = GradientBoostingRegressor(**gbm_cv_model_best_params_).fit(X_train,y_train)
y_pred = gbm_tuned.predict(X_test)
gbm_final = np.sqrt(mean_squared_error(y_test, y_pred))
print(gbm_final)


import pickle

pickle.dump(gbm_tuned, open('regression_model.pkl','wb'))

print("Model Kaydedildi")




















