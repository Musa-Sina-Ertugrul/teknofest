import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from copy import deepcopy
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import List, Dict
import torch.nn.functional as F
from model import Model
app = FastAPI()
pipe = pipeline("ner", model="51la5/roberta-large-NER", aggregation_strategy="simple")
# Load tokenizers
tokenizer_word = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
tokenizer_word.padding_side = "left"
tokenizer_word.pad_token = tokenizer_word.eos_token
tokenizer_word.add_special_tokens({'pad_token': '[PAD]'})

# Load NER model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained models
model_1 = Model().to(device)
model_1.load_state_dict(torch.load("/home/musasina/Desktop/projects/teknofest/msnet/model_restart.pth",map_location=torch.device('cpu'))["model_state_dict"])
model_1.eval()

# Sentiment mapping
sentiment_map_model_1 = {1: "nötr", 0: "olumsuz", 2: "olumlu"}

class Item(BaseModel):
    text: str = Field(..., example="""Fiber 100mb SuperOnline kullanıcısıyım yaklaşık 2 haftadır @Twitch @Kick_Turkey gibi canlı yayın platformlarında 360p yayın izlerken donmalar yaşıyoruz.  Başka hiç bir operatörler bu sorunu yaşamazken ben parasını verip alamadığım hizmeti neden ödeyeyim ? @Turkcell """)

def get_word_sentiment(text:str)->torch.Tensor:
    tokens_word = tokenizer_word([text], max_length=128, padding="max_length", truncation=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_word = model_1(tokens_word["input_ids"].view(1,-1)).argmax(dim=1)
    return output_word

def remove_punc(text:str) -> str:
    import string
    text = text.replace("'"," ")
    for punc in string.punctuation:
        text = text.replace(punc,"")

    return text

@app.post("/predict/", response_model=Dict[str, List])
async def predict(item: Item):
    text = remove_punc(deepcopy(item.text))
    splitted_text = text.split(" ")
    entities = pipe(text)
    entity_list = []
    for entity_dict in entities:
        entity_list.append(entity_dict["word"])

    entities = pipe(text)
    for entity_dict in entities:
        entity_list.extend([*(str(entity_dict["word"]).split(" "))])
    entity_dict = {}
    for entity in entity_list:
        entity_dict[entity] = get_word_sentiment(entity + " - " + remove_punc(text))

    results = []
    for entity,sentiment in entity_dict.items():
        results.append({
            "entity": entity,
            "sentiment": sentiment_map_model_1[sentiment.cpu().item()],
        })
    
    return {
        "entity_list":entity_list,
        "results": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8181)