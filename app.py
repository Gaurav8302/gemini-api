from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

class InputText(BaseModel):
    text: str

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-hf-username/your-model-name")
model = AutoModelForSequenceClassification.from_pretrained("your-hf-username/your-model-name")

id2label = {
    0: "web_search",
    1: "set_timer",
    2: "get_time",
    3: "facts_asking",
    4: "general_talk",
    5: "close_app",
    6: "open_app",
    7: "play_music",
    8: "get_weather",
    9: "get_date",
    10: "translate",
    11: "control_media",
    12: "calendar_check",
    13: "read_notifications",
    14: "screenshot",
    15: "define_word",
    16: "take_note",
    17: "control_volume",
    18: "system_status",
    19: "get_news",
    20: "email_send",
    21: "open_file",
    22: "set_alarm"
}

@app.get("/")
def read_root():
    return {"message": "API is running!"}

@app.post("/predict")
def predict(input_data: InputText):
    inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    label = id2label.get(prediction, "unknown")
    
    # Simple command extraction (can later use Gemini here)
    return {"intent": label, "command": input_data.text.lower()}
