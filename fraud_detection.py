# -*- coding: utf-8 -*-
"""
created by: Kostis Paraskevoudis
date: 24/02/2020
checked on Python3.8

Please set up the correct path to the csv file at line 426 --> df = pd.read_csv("...")
Use:
train_and_test_classifier: for training and testing a classification algorithm (SVC, Extra Trees, Random Forests)
fit_and_visualize_clustering: to perform clustering and visualize results

see the functions' description for more information

"""

import pandas as pd
import numpy as np
import random
import time
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve,roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def pie_chart(perc):
  '''
  this function plots a pie chart given the percentages of each class in the
  dataset
  IN
  perc: a list with percentages of the two classes
  '''
  explode = (0, 0.2)
  labels_ = ['Fraudulent', 'Non Fraudulent']
  fig, ax = plt.subplots(figsize=(8,8))
  ax.pie(perc, explode=explode, labels=labels_, autopct='%1.4f%%',
        shadow=False, startangle=90)
  plt.title("Class percentages in the dataset")
  plt.show()


def exploratory_analysis(df):
  '''
  this function performs some exploratory analysis in a passed dataframe
  IN
  df: the pandas dataframe with the dataset
  '''
  fraud = len(df['Class'][df['Class'] == 1])
  non_fraud = len(df['Class'][df['Class'] == 0])
  print("Shape of dataframe: ", df.shape, "\n", "Total rows: ", len(df) , "\n", 
        "Missing values: ", df.isna().sum().any(), "\n" , 
        "Total fraudulent transactions: ", fraud, " --> ", 
        round(fraud*100/len(df),4), "% \n", 
        "Total non-fraudulent transactions: ", non_fraud, " --> ",
        round(non_fraud*100/len(df),4), "%")
  pie_chart([round(fraud*100/len(df),4),round(non_fraud*100/len(df),4)])
  print("---------------------------------------")
  print("Columns and data types: ", df.info())
  print("---------------------------------------")
  print(df.describe())

def plot_features(df, class_name, column1, column2):
  '''
  function to plot two features wrt to the class
  IN
  df: pandas dataframe of dataset
  class_name: the target column name (string)
  column1: first feature column
  column2: second feature column
  '''
  fig, ax = plt.subplots(figsize=(10,10))
  plt.scatter(df[column1][df[class_name] == 0], 
              df[column2][df[class_name] == 0], 
              label="non fraudulent", alpha=0.5, linewidth=0.15, c='limegreen')
  plt.scatter(df[column1][df[class_name] == 1], 
              df[column2][df[class_name] == 1], 
              label="fraudulent", alpha=0.5, linewidth=0.15, c='purple')
  plt.legend()
  plt.axis('off')
  plt.title("2D representation of Data \n Columns: " + str(column1) + "and" + 
            str(column2))
  plt.show()

def class_report(cr, cf, fpr, tpr, score,feature_importance_normalized,columns):
  '''
  function to plot the classification report (precision, recall etc.) 
  on the test set, the confusion matrix, the roc curve with auc score and 
  importances

  IN:
  cr: classification report
  cf: confusion matrix
  fpr: false positive rates for all thresholds
  tpr: true positive rates for all thresholds
  feature_importance_normalized: the normalized feature importances
  columns: feature columns
  '''
  cr1 = cr
  cr1 = cr1.drop(['support'],axis = 1)
  cr1.index = ['Non Fraud','Fraud','Accuracy','Macro avg','Micro Avg']
  ax = sns.heatmap(cr1,annot=True,cmap="YlGnBu",fmt="f")
  plt.title("Classification report")
  plt.show()

  plt.close()
  conf_matr = pd.DataFrame(cf, columns = ['Predicted Non Fraud','Predicted Fraud'])
  conf_matr.index = ['Actual Non Fraud','Actual Fraud']
  ax = sns.heatmap(conf_matr,annot=True,cmap="YlGnBu",fmt=".0f")
  plt.title("Confusion Matrix")
  plt.show()
  plt.close()
  
  plt.plot(fpr,tpr,marker='.')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(" ROC curve \n AUC score: " + str(round(score,2)) )
  plt.show()
  plt.close()

  plt.figure(figsize=(12,12))
  plt.bar(columns, feature_importance_normalized) 
  plt.xlabel('Feature Labels') 
  plt.ylabel('Normalized Feature Importances') 
  plt.title('Feature Importances') 
  plt.show()

def hist_correlation_analysis(df):
  '''
  features histograms and correlation analysis
  IN
  df: pandas dataframe of dataset 
  '''
  df.hist(bins=50, figsize=(20,15))
  plt.show()
  f = plt.figure(figsize=(19, 15))
  plt.matshow(df.drop(["Class"],axis = 1).corr(), fignum=f.number)
  plt.xticks(range(df.drop(["Class"],axis = 1).select_dtypes(['number']).shape[1]), df.drop(["Class"],axis = 1).select_dtypes(['number']).columns, fontsize=14, rotation=45)
  plt.yticks(range(df.drop(["Class"],axis = 1).select_dtypes(['number']).shape[1]), df.drop(["Class"],axis = 1).select_dtypes(['number']).columns, fontsize=14)
  cb = plt.colorbar()
  cb.ax.tick_params(labelsize=14)
  plt.title('Correlation Matrix', fontsize=16)
  sns.pairplot(df)

def grid_search(df, class_name, model_name, param_grid, scoring):
  '''
  function to perfrom a grid search
  IN
  df: pandas dataframe of dataset
  class_name: the target column name (string)
  model_name: short name of the algorithm to be applied, accepted values are
    "rf", "Random Forests", "random forests" for RandomForestClassifier and
    "et", "Extra Trees", "extra trees" for ExtraTreesClassifier and
    "svc", "svm", "SVM", "SVC" for linear kernel Support Vector Classifier
  param_grid: dictionairy of hyperparameters to check 
              e.g. param_grid = {'max_depth':[ 5, 15, None], 
                                  'max_features': [None, 'sqrt'],
                                  'n_estimators':[100, 10],
                                  'min_samples_split':[2,3,5]}
                    for RandomForestClassifier
  scoring: metric to optimize ("recall", "precision")
  OUT
  cv: all GridSearchCV attributes
  cv.best_params: gridsearch optimal parameters
  '''

  #set target value and features for giving as input to GridSearch
  y = df[class_name]
  X= df.drop([class_name], axis = 1)

  #set up input of gridsearch
  if model_name in ["rf", "Random Forests", "random forests"]:
    model = RandomForestClassifier()
  elif model_name in ["et", "Extra Trees", "extra trees"]:
    model = ExtraTreesClassifier()
  elif model_name in ["svc", "svm", "SVM", "SVC"]:
    model = SVC()

  cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=scoring, n_jobs=-1)
  cv.fit(X, y)

  print(cv.cv_results_)

  return(cv, cv.best_params_)

def train_ensemble_classifier(df, class_name, models, smote=True, 
                              scaler_name = "standard",test_size = 0.3, 
                              save = False, voting = "hard"):
  '''
  function to apply scaling and SMOTE and then train an ensemble model 
  on a given dataframe. After training and predicting on test set, it calculates
  evaluation metrics and passes them to function class_report in order to 
  plot results
  IN
  df: pandas dataframe of dataset 
  class_name: the target column name (string)
  models: list with model names to include (must be of len 2 or 3) 
    accepted values are:
    "rf", "Random Forests", "random forests" for RandomForestClassifier and
    "et", "Extra Trees", "extra trees" for ExtraTreesClassifier and
    "svc", "svm", "SVM", "SVC" for linear kernel Support Vector Classifier
  smote: whether to use SMOTE for synthetic minority class Data generation 
  scaler_name: scaling technique to be applied ("standard" or "minmax")
  test_size: the proportion of test data to split
  save: whether to save the model to a .sav file for later use
  voting: how to generalize predictions into the final ("hard" or "soft")
          If ‘hard’, uses predicted class labels for majority rule voting. 
          Else if ‘soft’, predicts the class label based on the argmax of 
          the sums of the predicted probabilities, which is recommended for 
          an ensemble of well-calibrated classifiers.
  '''

  #train-test split
  y = df[class_name]
  X = df.drop([class_name], axis = 1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                      random_state=0)
  #set up the models
  models_lst=[]
  for model in models:
    if model in ["rf", "Random Forests", "random forests"]:
      models_lst.append(RandomForestClassifier())
    elif model_name in ["et", "Extra Trees", "extra trees"]:
      models_lst.append(ExtraTreesClassifier())
    elif model_name in ["svc", "svm", "SVM", "SVC"]:
      models_lst.append(SVC(probability=True))
  
  #set up the ensemble
  if len(models) == 2:
    ensemble_model = VotingClassifier(estimators=[('first', models_lst[0]), 
                                                  ('second', models_lst[1])], 
                                                  voting=voting)
  elif len(models) == 3:
    ensemble_model = VotingClassifier(estimators=[('first', models_lst[0]), 
                                                  ('second', models_lst[1]),
                                                  ('third', models_lst[2])], 
                                      voting=voting)
  if scaler_name in ["standard", "Standard", "standard scaler"]:
    scaler = StandardScaler()
  elif scaler_name in ["minmax", "MinMax", "MinMax Scaler"]:
    scaler = MinMaxScaler()
  if smote == True:  
    pipeline = Pipeline([('SMOTE', SMOTE()),('scaler', StandardScaler()), 
                       ('ensemble', ensemble_model)])
  else:
    pipeline = Pipeline([('scaler', StandardScaler()), 
                       ('ensemble', ensemble_model)])
  
  print("\nStarting training...\n")
  t = time.time()
  pipeline.fit(X_train, y_train)
  print("Finished Training in ", round(time.time() - t,2), "seconds")
  print("Starting predictions")
  t = time.time()
  y_pred = pipeline.predict(X_test)
  print("Finished ", len(y_pred), " predictions in ", round(time.time() - t,2), 
        "seconds")
  cr = pd.DataFrame(classification_report(y_test, 
                                        y_pred, digits=2,
                                        output_dict=True)).T
  cr['support'] = cr.support.apply(int)
  cf = confusion_matrix(y_true=y_test, y_pred=y_pred)
  print("\n-----------------------------------\n")
  print("Results of ", pipeline['model'], "\n")

  fpr, tpr, thresholds = roc_curve(y_test, pipeline.predict_proba(X_test)[:,1])
  score = roc_auc_score(y_pred,y_test)

  feature_importance = pipeline['model'].feature_importances_ 

  feature_importance_normalized = np.std([tree.feature_importances_ for tree in 
                                        pipeline['model'].estimators_], 
                                        axis = 0)
  
  class_report(cr, cf, fpr, tpr, score, feature_importance_normalized, 
                                        X_train.columns)
  print("\n-----------------------------------\n")
  if save == True:
    filename = "ensemble_finalized_model.sav"
    pickle.dump(pipeline, open(filename, 'wb'))

def train_and_test_classifier(df, class_name, model_name ="rf", 
                   smote=True, scaler_name = "standard",test_size = 0.3, 
                   save = False):
  '''
  function to apply scaling and SMOTE and then train an available model 
  on a given dataframe. After training and predicting on test set, it calculates
  evaluation metrics and passes them to function class_report in order to 
  plot results
  IN
  df: pandas dataframe of dataset 
  class_name: the target column name (string)
  model_name: short name of the algorithm to be applied, accepted values are
    "rf", "Random Forests", "random forests" for RandomForestClassifier and
    "et", "Extra Trees", "extra trees" for ExtraTreesClassifier and
    "svc", "svm", "SVM", "SVC" for linear kernel Support Vector Classifier
  smote: whether to use SMOTE for synthetic minority class Data generation 
  scaler_name: scaling technique to be applied ("standard" or "minmax")
  test_size: the proportion of test data to split
  save: whether to save the model to a .sav file for later use
  '''

  #train-test split
  y = df[class_name]
  X= df.drop([class_name], axis = 1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                      random_state=0)
 
  #pipeline setup
  if model_name in ["rf", "Random Forests", "random forests"]:
    model = RandomForestClassifier()
  elif model_name in ["et", "Extra Trees", "extra trees"]:
    model = ExtraTreesClassifier()
  elif model_name in ["svc", "svm", "SVM", "SVC"]:
    model = SVC(probability=True)
  if scaler_name in ["standard", "Standard", "standard scaler"]:
    scaler = StandardScaler()
  elif scaler_name in ["minmax", "MinMax", "MinMax Scaler"]:
    scaler = MinMaxScaler()

  if smote == True:
    pipeline = Pipeline([('SMOTE', SMOTE()),('scaler', scaler), 
                         ('model', model)])
  else:
    pipeline = Pipeline([('scaler', scaler),('model', model)])

  print("\nStarting training...\n")
  t = time.time()
  pipeline.fit(X_train, y_train)
  print("Finished Training in ", round(time.time() - t,2), "seconds")
  print("Starting predictions")
  t = time.time()
  y_pred = pipeline.predict(X_test)
  print("Finished ", len(y_pred), " predictions in ", round(time.time() - t,2), 
        "seconds")
  cr = pd.DataFrame(classification_report(y_test, 
                                        y_pred, digits=2,
                                        output_dict=True)).T
  cr['support'] = cr.support.apply(int)
  cf = confusion_matrix(y_true=y_test, y_pred=y_pred)
  print("\n-----------------------------------\n")
  print("Results of ", pipeline['model'], "\n")

  fpr, tpr, thresholds = roc_curve(y_test, pipeline.predict_proba(X_test)[:,1])
  score = roc_auc_score(y_pred,y_test)

  feature_importance = pipeline['model'].feature_importances_ 

  feature_importance_normalized = np.std([tree.feature_importances_ for tree in 
                                        pipeline['model'].estimators_], 
                                        axis = 0)
  
  class_report(cr, cf, fpr, tpr, score, feature_importance_normalized, 
                                        X_train.columns)
  print("\n-----------------------------------\n")
  if save == True:
    filename = str(model_name) + "_finalized_model.sav"
    pickle.dump(pipeline, open(filename, 'wb'))

def fit_and_visualize_clustering(df, class_column, column_1, column_2, 
                                 n_clusters):
  '''
  function to fit a MiniBatchKmeans and visualize clusters
  IN
  df: the pandas dataframe with the dataset
  class_column: the target column name (string)
  column_1: the first feature column to include in clustering
  column_2: the second feature column to include in clustering
  n_clusters: the number of clusters to apply on KMeans 
                (prefer 2 <= n_clusters <= 10 for performace reasons)
  '''
  features = df[[column_1, column_2]]
  scaler = StandardScaler()
  scaled_features = scaler.fit_transform(features)

  kmeans_mini = MiniBatchKMeans(n_clusters=n_clusters)
  print("\nStarting fitting...\n")
  t = time.time()
  kmeans_mini.fit(scaled_features)
  print("Finished Fitting in ", round(time.time() - t,2), "seconds")

  scaled_features_ = pd.DataFrame(scaled_features, 
                                  columns = [column_1,column_2])
  scaled_features_[class_column] = df[class_column]

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharex=True, 
                                 sharey=True)
  fig.suptitle("KMeans Clusters VS actual transaction types for V10 and V28 \n"
                        + str(n_clusters) + " Clusters", fontsize=16)

  #assign a random color to each cluster
  samples = random.sample(range(1, 100),
                          len(np.unique((kmeans_mini.labels_))))
  colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
  colors_list = list(colors.values())
  
  fte_colors = { x:colors_list[samples[x]]
                        for x in range(len(np.unique(kmeans_mini.labels_)))}
  km_colors = [fte_colors[label] for label in kmeans_mini.labels_]
 
  #plot clusters and real classes
  ax1.scatter(scaled_features[:, 0], scaled_features[:, 1], c=km_colors)
  ax1.legend()
  ax1.axis('off')
  ax2.scatter(scaled_features_[column_1][scaled_features_[class_column] == 0], 
              scaled_features_[column_2][scaled_features_[class_column] == 0], 
              label="non fraudulent", alpha=0.5, linewidth=0.15, c='#008fd5')
  ax2.scatter(scaled_features_[column_1][scaled_features_[class_column] == 1], 
              scaled_features_[column_2][scaled_features_[class_column] == 1], 
              label="fraudulent", alpha=0.5, linewidth=0.15, c="#fc4f30")
  ax2.legend()
  ax2.axis('off')
  ax1.title.set_text("Clusters from MiniBatch KMeans")
  ax2.title.set_text("Raw data and their actual class")
  plt.show()

df = pd.read_csv("/content/drive/MyDrive/creditcard.csv")       #replace with correct path of csv file
print(df.head())                                                #check if the data is loaded properly
#exploratory_analysis(df)                                       #uncomment if you wish to perform exploratory_analysis (null values, data types, correlations)
#hist_correlation_analysis(df)                                  #uncomment if you wish to see histograms of the features and their correlations
#plot_features(df, 'Class', 'V18', 'V9')                        #uncomment if you wish to plot two features wrt the class

#main classification function for both training and testing. see the function train_and_test_classifier for more
train_and_test_classifier(df,"Class", model_name ="et",
                                                 smote=True,
                                                 scaler_name = "standard", 
                                                 test_size = 0.3, save =True)


#fit_and_visualize_clustering(df, "Class", "V10", "V28", n_clusters = 7)   #uncomment if you wish to perform KMeans clustering. see function fit_and_visualize_clustering for more.

