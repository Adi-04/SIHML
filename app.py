from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import FastAPI
import pickle
from fastapi.middleware.cors import CORSMiddleware
import joblib
# import numpy as np
import collections
import pandas as pd
import uvicorn
import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# List of all possible symptoms
ALL_SYMPTOMS = ['itching',
                'nodal_skin_eruptions',
                'chills',
                'stomach_pain',
                'muscle_wasting',
                'vomiting',
                'spotting_ urination',
                'fatigue',
                'weight_loss',
                'breathlessness',
                'dark_urine',
                'pain_behind_the_eyes',
                'constipation',
                'abdominal_pain',
                'diarrhoea',
                'yellowing_of_eyes',
                'chest_pain',
                'fast_heart_rate',
                'dizziness',
                'excessive_hunger',
                'slurred_speech',
                'knee_pain',
                'muscle_weakness',
                'unsteadiness',
                'bladder_discomfort',
                'internal_itching',
                'muscle_pain',
                'altered_sensorium',
                'red_spots_over_body',
                'abnormal_menstruation',
                'increased_appetite',
                'lack_of_concentration',
                'receiving_blood_transfusion',
                'stomach_bleeding',
                'distention_of_abdomen',
                'blood_in_sputum',
                'prominent_veins_on_calf',
                'blackheads',
                'small_dents_in_nails',
                'blister']


# model = pickle.load()
model = joblib.load('./model/model.pkl')

# Initialize the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # Replace with your front-end's origin
    allow_origins=["http://localhost:8100",
                   "https://internal-sih-xi.vercel.app", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_conversational_chain():
    prompt_template = """
   Please provide detailed information about the disease Disease . The information should include:

   Causes: Explain the underlying causes or risk factors associated with this disease.
   Symptoms: List the common and significant symptoms that patients might experience.
   Diagnosis Methods: Describe the medical tests, exams, or procedures commonly used to diagnose this disease.
   Available Treatments: Outline the standard treatment options, including medications, therapies, or surgical interventions.
   Potential Complications: Discuss any serious or long-term complications that may arise if the disease is left untreated or poorly managed.
   Recommended Lifestyle Changes: Suggest lifestyle modifications that can help prevent or manage this disease effectively.
 
    Context:\n {context}?\n
    Disease: \n{user_disease}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=[
                            "context", "user_disease"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_disease):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_disease)
    chain = get_conversational_chain()

    response = chain.invoke(
        {"input_documents": docs, "user_disease": user_disease}, return_only_outputs=True)
    # print(response)
    return response


@app.post("/get_disease_info/")
async def get_disease_info(request: dict):
    # print(request)
    user_disease = request["user_disease"]
    return user_input(user_disease)


@app.post("/predict/")
async def predict(data: dict):

    # print(data)
    # print(type(data))
    input_features = {}

    for symptom in ALL_SYMPTOMS:
        if symptom in data.keys():
            input_features[symptom] = 1
        else:
            input_features[symptom] = 0
    l = pd.DataFrame(input_features, index=[0])
    # print(l)
    # print((model))
    prediction = model.predict(l)

    # Return the prediction result
    return {"prediction": prediction[0]}

@app.post("/priority/")
async def priority(data: list):
    # class CustomFeatureEngineering(BaseEstimator, TransformerMixin):
    #     def fit(self, X, y=None):
    #         self.transplant_categories_ = ['Kidney', 'heart', 'heart-kidney', 'heart-lungs', 'lung', 'lung-kidney', 'nothing']
    #         return self
        
    #     def transform(self, X):
    #         X_encoded = X.copy()
            
    #         # Ensure correct mapping
    #         X_encoded['Gender'] = X_encoded['Gender'].map({'Female': -1, 'Male': 1})
            
    #         # Feature engineering
    #         X_encoded['Heart_Condition_Severity_Index'] = (
    #             X_encoded['heart Attack'] * 100 +
    #             X_encoded['Heart Valve'] * 10 +
    #             X_encoded['Heart Defect at birth'] * 50 +
    #             X_encoded['Cardiomyopathy'] * 75
    #         )
            
    #         X_encoded['Lung_Condition_Severity_Index'] = (
    #             X_encoded['copd(lung_Disease)'] * 60 +
    #             X_encoded['Severe cystic fibrosis'] * 80
    #         )
            
    #         X_encoded['Kidney_Condition_Severity_Index'] = (
    #             X_encoded['kidney stones'] * 20 +
    #             X_encoded['Repeated urinary infections'] * 30 +
    #             X_encoded['Urinary Tract Infection'] * 40
    #         )
            
    #         X_encoded['Chronic_Condition_Severity_Index'] = (
    #             X_encoded['Heart_Condition_Severity_Index'] +
    #             X_encoded['Lung_Condition_Severity_Index'] +
    #             X_encoded['Kidney_Condition_Severity_Index'] +
    #             X_encoded['Diabities'] * 50
    #         )
            
    #         X_encoded['Age_Heart_Interaction'] = X_encoded['Age'] * X_encoded['Heart_Condition_Severity_Index']
    #         X_encoded['Age_Lung_Interaction'] = X_encoded['Age'] * X_encoded['Lung_Condition_Severity_Index']
    #         X_encoded['Age_Kidney_Interaction'] = X_encoded['Age'] * X_encoded['Kidney_Condition_Severity_Index']
    #         X_encoded['Age_Chronic_Interaction'] = X_encoded['Age'] * X_encoded['Chronic_Condition_Severity_Index']
            
    #         X_encoded['Gender_Heart_Interaction'] = X_encoded['Gender'] * X_encoded['Heart_Condition_Severity_Index']
    #         X_encoded['Gender_Kidney_Interaction'] = X_encoded['Gender'] * X_encoded['Kidney_Condition_Severity_Index']
    #         X_encoded['Gender_Lung_Interaction'] = X_encoded['Gender'] * X_encoded['Lung_Condition_Severity_Index']
            
    #         symptom_columns = ['heart Attack', 'Heart Valve', 'Heart Defect at birth', 'Cardiomyopathy', 
    #                         'Severe cystic fibrosis', 'copd(lung_Disease)', 'Repeated urinary infections', 
    #                         'Diabities', 'kidney stones', 'Urinary Tract Infection']
    #         X_encoded['Symptom_Count'] = X_encoded[symptom_columns].sum(axis=1)
            
    #         # One-hot encoding of 'Transplant' column while preserving NaN values
    #         X_encoded["Transplant"] = X_encoded["Transplant"].fillna("nothing")
    #         one_hot_encoded = pd.get_dummies(X_encoded['Transplant'])
    #         one_hot_encoded = one_hot_encoded.astype(int)
            
    #         # Ensure all predefined columns are present, filling missing ones with 0
    #         for category in self.transplant_categories_:
    #             if category not in one_hot_encoded.columns:
    #                 one_hot_encoded[category] = 0
            
    #         # Merging the one-hot encoded columns back into the dataframe
    #         X_encoded = pd.concat([X_encoded, one_hot_encoded], axis=1)
    #         X_encoded = X_encoded.drop(columns='Transplant')
            
    #         return X_encoded

    # # Define the columns for scaling
    # scale_columns = ['Age', 'Heart_Condition_Severity_Index', 'Lung_Condition_Severity_Index',
    #                 'Kidney_Condition_Severity_Index', 'Chronic_Condition_Severity_Index',
    #                 'Age_Heart_Interaction', 'Age_Lung_Interaction', 'Age_Kidney_Interaction',
    #                 'Age_Chronic_Interaction', 'Gender_Heart_Interaction', 'Gender_Kidney_Interaction',
    #                 'Gender_Lung_Interaction', 'Symptom_Count']

    # non_scaled_columns = ['Gender', 'Kidney', 'heart', 'heart-kidney', 'heart-lungs', 'lung', 'lung-kidney', 'nothing']

    # # Drop columns only if they exist
    # def drop_columns_if_exist(X):
    #     # Convert column names to strings to avoid the TypeError
    #     df = pd.DataFrame(X, columns=[str(col) for col in (scale_columns + non_scaled_columns)])
        
    #     columns_to_drop = ['heart Attack', 'Heart Valve', 'Heart Defect at birth', 'Cardiomyopathy', 
    #                     'Severe cystic fibrosis', 'copd(lung_Disease)', 'Repeated urinary infections', 
    #                     'Diabities', 'kidney stones', 'Urinary Tract Infection']
        
    #     # Drop columns only if they exist in the dataframe
    #     return df.drop(columns=[col for col in columns_to_drop if col in df.columns])


    # # Load the pipeline model
    # with open('ML_BACKEND\model\pipeline.pkl', 'rb') as f:
    #     pipeline = pickle.load(f)
    
    
    def calcAge(obj):
        dob = obj["dateOfBirth"]
        today = datetime.datetime.today()
        age = today.year - int(dob.split('-')[0])
        return age

    df = collections.defaultdict(list)

    for object in data:
        df["Age"].append(calcAge(object))
        df["Gender"].append(object["gender"])
        df['heart Attack'].append(object["conditions"]["heartAttack"])
        df['Heart Valve'].append(object["collections"]["heartValve"])
        df['Heart Defect at birth'].append(object["collections"]["heartDefectAtBirth"])
        df['Cardiomyopathy'].append(object["collections"]["cardiomyopathy"])
        df['Severe cystic fibrosis'].append(object["collections"]["severeCysticFibrosis"])
        df['copd(lung_Disease)'].append(object["collections"]["copd"])
        df['Repeated urinary infections'].append(object["collections"]["repeatedUrinaryInfections"])
        df['Diabities'].append(object["collections"]["diabities"])
        df['kidney stones'].append(object["collections"]["kidneyStones"])
        df['Urinary Tract Infection'].append(object["collections"]["urinaryTractInfection"])
        df['Transplant'].append(object["organNeeded"])
    
    df = pd.DataFrame(data=df)
    
    print(df)
    
    # pipeline = joblib.load("ML_BACKEND\model\pipeline.pkl")
    
    # prediction_probs = [x[1] for x in pipeline.predict_proba(df)]
    # emails = [x["emailAddress"] for x in data]
    
    # final_list = list(zip(emails, prediction_probs))

    # Return the prediction result
    # return final_list
    return 0


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
