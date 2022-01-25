
# 1. Library imports
#import uvicorn
from fastapi import FastAPI
#from Model import CreditModel
from pydantic import BaseModel
import pandas as pd
import joblib
# 2. Create app and model objects
app = FastAPI()
#model = CreditModel()
df = pd.read_csv('data.csv')
model_fname_ = 'random_model.pkl'
model = joblib.load(model_fname_)
class CreditModel(BaseModel) :
    index : int

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')
def predict_credit(credit: CreditModel):
    data = credit.dict()
    index = int(data["index"])
    client = df.iloc[[index]]
    preds = model.predict_proba(client)
    if preds[0][0] > preds[0][1]:
        prediction = "Solvable"
        probability = preds[0][0]
    else:
        prediction = " Non Solvable"
        probability = preds[0][1]

    #prediction, probability = model.predict(data)
    return {
        'prediction': prediction,
        'probability': probability
    }


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
#if __name__ == '__main__':
  #  uvicorn.run(app, host='127.0.0.1', port=8000)
