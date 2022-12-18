from fastapi import FastAPI
from fastapi import HTTPException

from brain import analyze

app = FastAPI()


@app.get('/')
async def get_root():
    return {"message" : "hello world"}



@app.post("/detect/")
async def get_emotion(prompt:str = ""):

    emotions = await analyze.predict(prompt)
    return emotions