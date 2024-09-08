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
import pandas as pd
import uvicorn

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


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 4000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
