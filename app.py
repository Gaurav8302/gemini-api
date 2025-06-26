import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-pro")

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "Intent API running with Gemini"}

@app.post("/intent")
def get_intent(data: InputText):
    prompt = f"""
    Detect the intent of this message in one word from the following:
    web_search, set_timer, get_time, facts_asking, general_talk, close_app, open_app, play_music, get_weather, get_date,
    translate, control_media, calendar_check, read_notifications, screenshot, define_word, take_note, control_volume,
    system_status, get_news, email_send, open_file, set_alarm.

    Return ONLY the intent word.

    Message: {data.text}
    Intent:"""

    response = model.generate_content(prompt)
    return {"intent": response.text.strip().lower()}

@app.post("/action")
def extract_command(data: InputText):
    prompt = f"""
    Convert this user message into a clear action command. Focus on whether it's about opening or closing an app, or performing another task.

    Examples:
    - "please open whatsapp" -> open whatsapp
    - "launch notepad" -> open notepad
    - "terminate chrome" -> close chrome
    - "what's the time" -> get time
    - "define inertia" -> define inertia

    Message: {data.text}
    Command:
    """
    response = model.generate_content(prompt)
    return {"command": response.text.strip().lower()}
