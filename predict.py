import pickle
from typing import Dict, Any
from fastapi import FastAPI

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal, Optional


# Define the model using Pydantic's BaseModel
class Citizen(BaseModel):
    """
    Pydantic model for Telco Citizen Data, enforcing known categorical values
    and setting constraints on numerical features.
    """
    workclass: Literal[
        "Private", "Self-emp-not-inc", "Local-gov", "?", "State-gov",
        "Self-emp-inc", "Federal-gov", "Without-pay", "Never-worked"
    ]
    education: Literal[
        "HS-grad", "Some-college", "Bachelors", "Masters", "Assoc-voc", "11th",
        "Assoc-acdm", "10th", "7th-8th", "Prof-school", "9th", "12th", "Doctorate",
        "5th-6th", "1st-4th", "Preschool"
    ]
    marital_status: Literal[
        "Married-civ-spouse", "Never-married", "Divorced", "Separated",
        "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ]
    occupation: Literal[
        "Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical", "Sales",
        "Other-service", "Machine-op-inspct", "?", "Transport-moving", "Handlers-cleaners",
        "Farming-fishing", "Tech-support", "Protective-serv", "Priv-house-serv", "Armed-Forces"
    ]
    relationship: Literal[
        "Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"
    ]
    race: Literal[
        "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
    ]
    sex: Literal["Male", "Female"]
    native_country: Literal[
        "United-States", "Mexico", "?", "Philippines", "Germany", "Canada", "Puerto-Rico",
        "El-Salvador", "India", "Cuba", "England", "Jamaica", "South", "China", "Italy",
        "Dominican-Republic", "Vietnam", "Guatemala", "Japan", "Poland", "Columbia", "Taiwan",
        "Haiti", "Iran", "Portugal", "Nicaragua", "Peru", "France", "Greece", "Ecuador",
        "Ireland", "Hong", "Trinadad&Tobago", "Cambodia", "Thailand", "Laos", "Yugoslavia",
        "Outlying-US(Guam-USVI-etc)", "Honduras", "Hungary", "Scotland", "Holand-Netherlands"
    ]
    income: Literal["<=50K", ">50K"]
    
    age: int = Field(..., ge=17, le=90)
    fnlwgt: int = Field(..., ge=12285, le=1484705)
    education_num: int = Field(..., ge=1, le=16)
    capital_gain: int = Field(..., ge=0, le=99999)
    capital_loss: int = Field(..., ge=0, le=4356)
    hours_per_week: int = Field(..., ge=1, le=99)



#response model

class PredictResponse(BaseModel):
    income_probability: float
    income: bool


app = FastAPI(title="income-prediction")

with open('model.bin', 'rb') as f_in:
   pipeline = pickle.load(f_in)
 
# X= dv.transform(citizen)

def predict_single(citizen):
    result = pipeline.predict_proba(citizen)[0, 1]
    return float(result)

@app.get("/predict")
def predict(citizen: Citizen) -> PredictResponse:
     
    prob = predict_single(citizen.dict())
    
    return PredictResponse (
        income_probability=prob,
        income=bool(prob >= 0.5)
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)