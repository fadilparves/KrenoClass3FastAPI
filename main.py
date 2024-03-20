import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union

class LoanApplication(BaseModel):
    Loan_ID: Union[str, None]
    Gender: Union[str, None]
    Married: Union[str, None]
    Dependents: Union[str, None]
    Education: Union[str, None]
    Self_Employed: Union[str, None]
    ApplicantIncome: Union[int, None]
    CoapplicantIncome: Union[float, None]
    LoanAmount: Union[float, None]
    Loan_Amount_Term: Union[float, None]
    Credit_History: Union[float, None]
    Property_Area: Union[str, None]

app = FastAPI()

@app.get("/")
async def root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(request: LoanApplication):
    data = request.model_dump()
    df = pd.DataFrame([data])
    
    # fill empty values for Credit_History column with value 1
    df['Credit_History'].fillna(1, inplace=True)
    df['Self_Employed'].fillna('No', inplace=True)
    df['Dependents'].fillna("0", inplace=True)
    df['Gender'].fillna("Male", inplace=True)

    # remove any data that is still has null value
    df = df.dropna()
    
    if df.empty:
        return {"error": "Data is not valid format"}
    
    # data processing and mapping
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0) 
    df['Married'] = df['Married'].apply(lambda x: 1 if x == 'Married' else 0)
    df['Dependents'] = df['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3}) # domain knowledge feature
    df['Education'] = df['Education'].apply(lambda x: 1 if x == 'Graduate' else 0)
    df['Self_Employed'] = df['Self_Employed'].apply(lambda x: 1 if x == 'No' else 0)
    df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Rural': 0, 'Semiurban': 1}) #ordinal feature

    # load model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    loan_id = df.iloc[0]['Loan_ID']
    input = df.drop(columns=['Loan_ID'])
    prediction = model.predict(input)
    prediction = prediction.tolist()
    
    return {"loan_id": loan_id, 
            "predicted_class": prediction[0], 
            "predicted_class_name": "Approve" if prediction[0] == 1 else "Reject"}