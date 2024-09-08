import pickle
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from datetime import datetime
import collections


class CustomFeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.transplant_categories_ = ['Kidney', 'heart', 'heart-kidney', 'heart-lungs', 'lung', 'lung-kidney', 'nothing']
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        
        # Ensure correct mapping
        X_encoded['Gender'] = X_encoded['Gender'].map({'Female': -1, 'Male': 1})
        
        # Feature engineering
        X_encoded['Heart_Condition_Severity_Index'] = (
            X_encoded['heart Attack'] * 100 +
            X_encoded['Heart Valve'] * 10 +
            X_encoded['Heart Defect at birth'] * 50 +
            X_encoded['Cardiomyopathy'] * 75
        )
        
        X_encoded['Lung_Condition_Severity_Index'] = (
            X_encoded['copd(lung_Disease)'] * 60 +
            X_encoded['Severe cystic fibrosis'] * 80
        )
        
        X_encoded['Kidney_Condition_Severity_Index'] = (
            X_encoded['kidney stones'] * 20 +
            X_encoded['Repeated urinary infections'] * 30 +
            X_encoded['Urinary Tract Infection'] * 40
        )
        
        X_encoded['Chronic_Condition_Severity_Index'] = (
            X_encoded['Heart_Condition_Severity_Index'] +
            X_encoded['Lung_Condition_Severity_Index'] +
            X_encoded['Kidney_Condition_Severity_Index'] +
            X_encoded['Diabities'] * 50
        )
        
        X_encoded['Age_Heart_Interaction'] = X_encoded['Age'] * X_encoded['Heart_Condition_Severity_Index']
        X_encoded['Age_Lung_Interaction'] = X_encoded['Age'] * X_encoded['Lung_Condition_Severity_Index']
        X_encoded['Age_Kidney_Interaction'] = X_encoded['Age'] * X_encoded['Kidney_Condition_Severity_Index']
        X_encoded['Age_Chronic_Interaction'] = X_encoded['Age'] * X_encoded['Chronic_Condition_Severity_Index']
        
        X_encoded['Gender_Heart_Interaction'] = X_encoded['Gender'] * X_encoded['Heart_Condition_Severity_Index']
        X_encoded['Gender_Kidney_Interaction'] = X_encoded['Gender'] * X_encoded['Kidney_Condition_Severity_Index']
        X_encoded['Gender_Lung_Interaction'] = X_encoded['Gender'] * X_encoded['Lung_Condition_Severity_Index']
        
        symptom_columns = ['heart Attack', 'Heart Valve', 'Heart Defect at birth', 'Cardiomyopathy', 
                           'Severe cystic fibrosis', 'copd(lung_Disease)', 'Repeated urinary infections', 
                           'Diabities', 'kidney stones', 'Urinary Tract Infection']
        X_encoded['Symptom_Count'] = X_encoded[symptom_columns].sum(axis=1)
        
        # One-hot encoding of 'Transplant' column while preserving NaN values
        X_encoded["Transplant"] = X_encoded["Transplant"].fillna("nothing")
        one_hot_encoded = pd.get_dummies(X_encoded['Transplant'])
        one_hot_encoded = one_hot_encoded.astype(int)
        
        # Ensure all predefined columns are present, filling missing ones with 0
        for category in self.transplant_categories_:
            if category not in one_hot_encoded.columns:
                one_hot_encoded[category] = 0
        
        # Merging the one-hot encoded columns back into the dataframe
        X_encoded = pd.concat([X_encoded, one_hot_encoded], axis=1)
        X_encoded = X_encoded.drop(columns='Transplant')
        
        return X_encoded

# Define the columns for scaling
scale_columns = ['Age', 'Heart_Condition_Severity_Index', 'Lung_Condition_Severity_Index',
                 'Kidney_Condition_Severity_Index', 'Chronic_Condition_Severity_Index',
                 'Age_Heart_Interaction', 'Age_Lung_Interaction', 'Age_Kidney_Interaction',
                 'Age_Chronic_Interaction', 'Gender_Heart_Interaction', 'Gender_Kidney_Interaction',
                 'Gender_Lung_Interaction', 'Symptom_Count']

non_scaled_columns = ['Gender', 'Kidney', 'heart', 'heart-kidney', 'heart-lungs', 'lung', 'lung-kidney', 'nothing']

# Drop columns only if they exist
def drop_columns_if_exist(X):
    # Convert column names to strings to avoid the TypeError
    df = pd.DataFrame(X, columns=[str(col) for col in (scale_columns + non_scaled_columns)])
    
    columns_to_drop = ['heart Attack', 'Heart Valve', 'Heart Defect at birth', 'Cardiomyopathy', 
                       'Severe cystic fibrosis', 'copd(lung_Disease)', 'Repeated urinary infections', 
                       'Diabities', 'kidney stones', 'Urinary Tract Infection']
    
    # Drop columns only if they exist in the dataframe
    return df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Data Preprocessing
def preprocess(df):
    # Step 1: Custom Feature Engineering
    custom_feature_engineer = CustomFeatureEngineering()
    
    train = pd.read_csv("D:\\LifeDevice_SIH\\ML_BACKEND\\model\\Train.csv")
    X_train = train.drop("Needed_or_not", axis=1)
    y_train = train["Needed_or_not"]
    
    custom_feature_engineer.fit(X_train, y_train)
    X_transformed = custom_feature_engineer.transform(df)

    # Step 2: Scaling columns
    scaler = MinMaxScaler()
    X_scaled = X_transformed.copy()

    # Apply scaling only to the specific columns
    X_scaled[scale_columns] = scaler.fit_transform(X_transformed[scale_columns])

    # Non-scaled columns remain unchanged
    X_scaled[non_scaled_columns] = X_transformed[non_scaled_columns]

    # Step 3: Drop unwanted columns if they exist
    X_final = drop_columns_if_exist(X_scaled)

    return X_final


def calcAge(obj):
        dob = obj["dateOfBirth"]
        today = datetime.today()
        age = today.year - int(dob.split('-')[0])
        return age

def predict_pipeline(data: dict) -> list:
    df = collections.defaultdict(list)

    for object in data.values():
        df["Age"].append(calcAge(object))
        df["Gender"].append(object["gender"])
        df['heart Attack'].append(object["conditions"]["heartAttack"])
        df['Heart Valve'].append(object["conditions"]["heartValve"])
        df['Heart Defect at birth'].append(object["conditions"]["heartDefectAtBirth"])
        df['Cardiomyopathy'].append(object["conditions"]["cardiomyopathy"])
        df['Severe cystic fibrosis'].append(object["conditions"]["severeCysticFibrosis"])
        df['copd(lung_Disease)'].append(object["conditions"]["copd"])
        df['Repeated urinary infections'].append(object["conditions"]["repeatedUrinaryInfections"])
        df['Diabities'].append(object["conditions"]["diabities"])
        df['kidney stones'].append(object["conditions"]["kidneyStones"])
        df['Urinary Tract Infection'].append(object["conditions"]["urinaryTractInfection"])
        df['Transplant'].append(object["organNeeded"])
    
    df = pd.DataFrame(data=df)
    final_df = preprocess(df)
    
    # with open('D:\\LifeDevice_SIH\\ML_BACKEND\\model\\model.pkl', 'rb') as file:
    #     model = pickle.load(file)
    model = joblib.load('D:\\LifeDevice_SIH\\ML_BACKEND\\model\\model_final.pkl')
    
    prediction_probs = [x[1] for x in model.predict_proba(final_df)]
    emails = [x["emailAddress"] for x in data.values()]
    
    final_list = list(zip(emails, prediction_probs))

    # Return the prediction result
    return final_list
    
