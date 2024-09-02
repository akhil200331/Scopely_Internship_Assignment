# importing necessary libraries
import numpy as np
import pandas as pd # This library is to work with dataframes
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import AdaBoostClassifier
from src.update_metrics import update
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from src.box_plotting1 import box_plotting
# from sklearn.ensemble import StackingClassifier
import shap
import pickle

# DATA LOADING

# Reading csv file
data=pd.read_csv("D:/SCOPELY INTERNSHIP ASSIGNMENT/user churn prediction/data/game_user_churn.csv")

# print(data.head()) #used to display first 5 rows or tuples


# As we know the user-ID is not at all useful in prediction
# So we are dropping that column
data=data.drop("userID",axis=1)
print(data.head())

# checking for null values (DATA CLEANING)

print(data.isnull().sum())
#from above output we clearly see that no null values are there

#Statistical Summary of the data
print(data.describe())

# One hot encoding
label_encoder=LabelEncoder()

l=["gender","country","game_genre","subscription_status","device_type","favorite_game_mode"]
for i in l:
  data[i]=label_encoder.fit_transform(data[i])


#Data Visualisation

# Plot the heatmap for corrlation
plt.figure(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", cbar=False, square=True,annot_kws={"rotation": 45,"size":4})
plt.title("Correlation of Features")
plt.show()

#Count plot
sns.countplot(data=data,x="churn")# here we came to know that given dataset is imbalanced or not if imbalanced we have to do over or undersampling using smote technique


#Box plot for outlier detection
box_plotting(data,10,2,(20,40))



# DATA NORMALIZATION
sc=StandardScaler()
dc=list(data.columns)
dc.remove("churn")
for i in l:
  dc.remove(i)
data_scaled=pd.DataFrame(sc.fit_transform(data.drop(["gender","country","game_genre","subscription_status","device_type","favorite_game_mode","churn"],axis=1)),columns=dc)
for i in l:
    data_scaled[i]=data[i]
data_scaled["churn"]=data["churn"]
# data_scaled.head()




#Train Test split
X=data_scaled.drop(["churn"],axis=1)
Y=data_scaled["churn"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


#as the given data is imbalanced we can clearly seen that 2500 1 and 7500 0
# so we do oversampling which means we increase the samples to balance the minority class
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, Y_train)


#MODEL TRAINING

# dict for storing accuracy and other metrics
df=pd.DataFrame(columns=["Model_Name","Accuracy","Recall","Precision","F1-score"]) #Simply Creating DataFrame
# d={}

# ----------------------1. Random Forest---------------------
rf=RandomForestClassifier(criterion="gini",max_depth=400,n_estimators=30)
rf.fit(X_train_smote,y_train_smote) #Fitting the model
rfy=rf.predict(X_test)  #predicting using random forest
print("Random Forest Confusion Matrix:",confusion_matrix(Y_test,rfy))

#If we want to show confusion mtrix in plot using heatmap
#sns.heatmap(confusion_matrix(Y_test,rfy),annot=True)

# d["Random _forest"]=[accuracy_score(Y_test, rfy),recall_score(Y_test, rfy),precision_score(Y_test, rfy),f1_score(Y_test, rfy)]
df=update(Y_test,rfy,df,"Random Forest")


# ----------------------2. LightGBM---------------------
train_data = lgb.Dataset(X_train_smote, label=y_train_smote)
test_data = lgb.Dataset(X_test, label=Y_test, reference=train_data)


# Set parameters
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',  # Use 'multiclass' for multi-class classification
    'metric': 'binary_logloss' # Use 'multi_logloss' for multi-class classification
}

# Train the model
lgbm_model = lgb.train(params, train_data)
y_pred = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)

# Convert probabilities to binary predictions
y_pred_binary = [1 if x >= 0.5 else 0 for x in y_pred]

print("LIGHTGBM Confusion Matrix:",confusion_matrix(Y_test,y_pred_binary))

#d["LIGHTGBM"]=[accuracy_score(Y_test, y_pred_binary),recall_score(Y_test, y_pred_binary),precision_score(Y_test, y_pred_binary),f1_score(Y_test, y_pred_binary)]
df=update(Y_test,y_pred_binary,df,"LIGHTGBM")

#-----------------3. KNN Classifier----------------
knn = KNeighborsClassifier(n_neighbors=4)

# Train the model
knn.fit(X_train_smote, y_train_smote)

# Predict on the test set
y_pred_knn = knn.predict(np.array(X_test))

print("KNN Confusion Matrix:",confusion_matrix(Y_test,y_pred_knn))

#d["KNN"]=[accuracy_score(Y_test, y_pred_knn),recall_score(Y_test, y_pred_knn),precision_score(Y_test, y_pred_knn),f1_score(Y_test, y_pred_knn)]
df=update(Y_test,y_pred_knn,df,"KNN")



#-----------------4. Naive Bayes-------------
nb = GaussianNB()
# Train the model
nb.fit(X_train_smote, y_train_smote)

# Predict on the test set
y_pred_nb = nb.predict(X_test)


print("Naive Confusion Matrix:",confusion_matrix(Y_test,y_pred_nb))

#d["Naive Bayes"]=[accuracy_score(Y_test, y_pred_nb),recall_score(Y_test, y_pred_nb),precision_score(Y_test, y_pred_nb),f1_score(Y_test, y_pred_nb)]
df=update(Y_test,y_pred_nb,df,"Naive Bayes")




#--------------5. Stacked Regressor(Base=Naive bayes,Final=KNN)-----------
base_estimators = [
    ('nb', GaussianNB()),
]

# Final estimator
final_estimator = KNeighborsClassifier(n_neighbors=4)

# Create a stacked classifier
stacking_clf = StackingClassifier(estimators=base_estimators, final_estimator=final_estimator)

# Train the stacked model
stacking_clf.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred = stacking_clf.predict(X_test)

print("STACKED Confusion Matrix:",confusion_matrix(Y_test,y_pred))

#d["STACKED"]=[accuracy_score(Y_test, y_pred),recall_score(Y_test, y_pred),precision_score(Y_test, y_pred),f1_score(Y_test, y_pred)]
df=update(Y_test,y_pred,df,"STACKED")



# Storing all the metric in models which we are designed
# df = pd.DataFrame(d)

# # Transpose the DataFrame
# transposed_df = df.T
# transposed_df.columns=["Accuracy","Recall","Precision","F1-score"]
print("---------------Data Frame with all metrics-------------")
print(df)




    
#Explainable AI
# Finally We can explain how the each and every attribute contributed in model building like it transparently visible to audience

# Use SHAP to explain predictions
background_data = shap.kmeans(X_train_smote, 100)  # Use k-means to summarize the background data into 100 clusters
explainer = shap.KernelExplainer(nb.predict_proba, background_data)
shap_values = explainer.shap_values(X_test)

# Plot the SHAP summary plot for the first class
shap.summary_plot(shap_values, X_test, feature_names=list(data.columns))

