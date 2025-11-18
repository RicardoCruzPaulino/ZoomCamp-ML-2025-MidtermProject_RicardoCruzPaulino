#!/usr/bin/env python
# coding: utf-8

 


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from imblearn.under_sampling import RandomUnderSampler
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mutual_info_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OneHotEncoder,
    LabelBinarizer,
    MultiLabelBinarizer
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import pickle


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')

 ### Models and Hyperparameter
 
models = {
'Logistic Regression': LogisticRegression(max_iter=1000)
,    'k-NN': KNeighborsClassifier()
,    'Decision Tree': DecisionTreeClassifier()
,'Random Forest': RandomForestClassifier(n_estimators=100)
,'XGBoost': XGBClassifier()
# ,    'SVM': SVC()
# ,    'MLP': MLPClassifier(max_iter=1000)
}


# Selected model after 21 trainingn process

models = {
'XGBoost': XGBClassifier()

}



# Define the parameter grid for GridSearchCV for each model
param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.1, 1.0, 10.0]
    },
    'k-NN': {
        'classifier__n_neighbors': [3, 5, 7]
    },
    'Decision Tree': {
        'classifier__max_depth': [None, 10, 20]
    },
    'Random Forest': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20]
    },

    'XGBoost': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__gamma': [0, 0.1]
        },
    'SVM': {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__kernel': ['linear', 'rbf']
    },
    'MLP': {
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'classifier__alpha': [0.0001, 0.001]
    }
}
 

###  Functions
def save_model (filename, model):
    with open(filename, 'wb') as f_out:
        pickle.dump(model, f_out)
    print(f'The model is saved to {filename}')

def evaluate_model(name, model, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}\n")

# Evaluation of the best model in the test set
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    #return accuracy, precision, recall, f1, roc_auc

#Utilities Classes and Functions

def age_binning(data):
  bins = np.linspace(min(data["age"]), max(data["age"]), 4)
  bins
  group_names = ['Young', 'Adult', 'Elder']

  data['age-binned'] = pd.cut(data['age'], bins, labels=group_names, include_lowest=True )
  data[['age','age-binned']].head(20)
  return data


def handle_missing_values(data):
  #Replace null valueswith NaN
  data.replace('?', np.nan, inplace=True)
  # Imputar valores perdidos en variables numéricas con la mediana
  numeric_imputer = SimpleImputer(strategy='median')
  data[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']] = numeric_imputer.fit_transform(data[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']])

  # Impute missing valuesin categorical variables using the mode
  categorical_imputer = SimpleImputer(strategy='most_frequent')
  data[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']] = categorical_imputer.fit_transform(data[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']])
  return data

def replace_outliers_with_median(data):
  for column in data.select_dtypes(include=['int64', 'float64']).columns:
    z_scores = stats.zscore(data[column])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3)
    data[column] = data[column].where(filtered_entries, data[column].median())
    
    return data
    
def binarize_categorical_features(df, categorical_features):
  for feature in categorical_features:
    print(feature)
    lb = LabelBinarizer()
    
    df_bin = lb.fit_transform(df[feature])
   # df_bin[]
    df_bin = pd.DataFrame(df_bin, columns=[f"{feature}_{i}" for i in range(df_bin.shape[1])])
    df = pd.concat([df, df_bin], axis=1)
    df.drop(feature, axis=1, inplace=True)
    return df

# Balancing Samples by Applying Undersampling to Training and Test Sets
def UnderSampling_data(X, y, sampling_strategy = 0.8, test_size=0.3):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  random_under_sampler = RandomUnderSampler(sampling_strategy=sampling_strategy)
  X_train_res, y_train_res = random_under_sampler.fit_resample(X_train, y_train)
  X_test_res, y_test_res = random_under_sampler.fit_resample(X_test, y_test)
  return  X_train_res,  X_test_res, y_train_res, y_test_res

def get_best_model(results_models):
  """
  This function selects the best model from a list of model results based on the test score.

  Args:
    results_models: A list of lists, where each inner list represents a model
      and contains [model_name, best_params, best_cv_score, test_score].

  Returns:
    A list containing the name and test score of the best model, or None if
    the input list is empty or invalid.
  """
  if not results_models or not all(isinstance(model_data, list) and len(model_data) == 4 for model_data in results_models):
    return None

  best_model_data = max(results_models, key=lambda x: x[3])  # Find the model with the highest test score

  return [best_model_data[0], best_model_data[1], best_model_data[2], best_model_data[3]] # Return the best model's name and test score




def load_data():
    url =  "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"


    column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                    "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    df = pd.read_csv(url, names=column_names, sep=',', engine='python')
       
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_columns:
        print(c)
        df[c] = df[c].str.lower().str.strip().str.replace(' ', '_')
    print(df["income"].unique())
    ### Binarize the target variable
    df["income"] = df["income"].apply(lambda x: 1 if x == '>50k' else 0)
    ### **Handle Missing Values**
    df = handle_missing_values(df)
    #*** Handling   Outliers Values *************************")
    df = replace_outliers_with_median(df)
    # Applying Binning to variable Age
    df = age_binning(df)
    print("Dataframe after preprocessing:")
    print(df.head())

    return df
 

def train_model(data, models, param_grids  ):
    numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = ['age-binned', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
        
    #Preproccesing
    X = data.drop("income", axis=1)
    y = data["income"] 

    print("Preparamos el Dataset Train")
 
    # Split data into training and test

    print("Applying the Undersampling function to balance the dataset")

    X_train, X_test, y_train, y_test = UnderSampling_data(X, y, sampling_strategy = 0.8, test_size=0.3 )
    
    results_models=[]
    fitted_grid_searches = {}
    
    
# Create the preprocessor with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

 

# Train and evaluate each model using GridSearchCV
    for model_name in models:

        pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ("classifier", models[model_name])
        ])

        grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=5)
        grid_search.fit(X_train, y_train)

        print(f"Model: {model_name}")
        print("Better parameters found: ", grid_search.best_params_)
        print("Best cross-validation score: ", grid_search.best_score_)

        # Evaluar el modelo en el conjunto de prueba
        test_score = grid_search.score(X_test, y_test)
        print("Score on the test set: ", test_score)
        print("\n")

        results_models.append([model_name, grid_search.best_params_, grid_search.best_score_, test_score])
        fitted_grid_searches[model_name] = grid_search # Store the fitted grid_search object

    # Create a DataFrame from results_models before saving to CSV
        #to_csv("results_models.csv", index=False)
        results_df = pd.DataFrame(results_models, columns=['Model', 'Best Parameters', 'CV Score', 'Test Score'])
        results_df.to_csv("results_models.csv", index=False) # Save the DataFrame to CSV
        
        best_model_info = get_best_model(results_models) # Use the renamed function
        if best_model_info:
            best_model_name, best_model_parameters, best_model_train_score ,  best_model_test_score = best_model_info
            print(f"El mejor modelo es: {best_model_name} con una puntuación en el conjunto de prueba de: {best_model_test_score}")

            # Retrieve the actual best trained model using the stored grid_search objects
            if best_model_name in fitted_grid_searches:
                best_trained_model = fitted_grid_searches[best_model_name].best_estimator_
                print(f"Best trained model object: {best_trained_model}")
            else:
                print("Error: Best trained model not found in fitted_grid_searches.")
                best_trained_model = None # Set to None if not found
        else:
            print("The best model could not be determined.")
            best_trained_model = None # Set to None if no best model info
    
    


        return  best_trained_model 




df = load_data()
pipeline = train_model (df, models, param_grids)
save_model('model.bin', pipeline)
