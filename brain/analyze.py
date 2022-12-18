from transformers import pipeline
import json


model = pipeline("sentiment-analysis", "j-hartmann/emotion-english-distilroberta-base")


async def predict(prompt: str):

    predictions = model(prompt, top_k=7)
    predictions = convert_list_of_dict_to_json(predictions)

    return {"prompt": prompt, "emotions": predictions}


def convert_list_of_dict_to_json(list_of_dict: list[dict]):

    new_dict = {}
    new_dict = {d["label"]: d["score"] for d in list_of_dict}

    return new_dict
