# from pydantic import BaseModel,Field
# from typing import Annotated
# import pickle
# from fastapi import FastAPI
# from fastapi.responses import JSONResponse

# # open model
# with open("sentiment.pkl", "rb") as f:
#   model = pickle.load(f)

# class UserInput(BaseModel):
#   comment : Annotated[str, Field(..., description="Enter your comments : ")]

# app = FastAPI()

# @app.get("/hello")
# def hello():
#   return "This is sentiment analysis API"

# @app.post("/predict")
# def predict(data : UserInput):
#   input_df = data.comment


#   try:

#     prediction = model(input_df)

#     return JSONResponse(status_code = 200, content = {"prediction":prediction})
#   except Exception as e:
#     return JSONResponse(status_code=500, content={"error": str(e)})


from pydantic import BaseModel, Field
from typing import Annotated
import pickle
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# ✅ preprocessing.py থেকে import
from preprocessing import Text_procecess

# ✅ তিনটা pkl ফাইল লোড করো
with open("sentiment.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("le.pkl", "rb") as f:
    le = pickle.load(f)


# ✅ সম্পূর্ণ prediction pipeline
def sentiment(text):
    cleaned = Text_procecess(text)
    vector = tfidf.transform([cleaned])       # TF-IDF vectorize
    prediction = model.predict(vector)[0]     # model predict
    return le.inverse_transform([prediction])[0]  # numeric → label


class UserInput(BaseModel):
    comment: Annotated[str, Field(..., description="Enter your comments : ")]


app = FastAPI()


@app.get("/hello")
def hello():
    return "This is sentiment analysis API"


@app.post("/predict")
def predict(data: UserInput):
    try:
        prediction = sentiment(data.comment)  # ✅ সঠিকভাবে call হচ্ছে
        return JSONResponse(status_code=200, content={"prediction": prediction})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})