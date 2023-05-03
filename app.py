import uvicorn
import requests
import logging
import os
from time import time, ctime
from typing import List, Dict, Annotated
from datetime import datetime, timezone
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from stt import Speech_to_Text

VOICE_PATH = 'voice'
SAVE_WEBM_PATH = "voice/webm_files"
SAVE_WAV_PATH = "voice/wav_files"
INTENT_ENTITY_URL = "http://localhost:8001/api/v1/intent_entity_classify"

origins = [
    "http://localhost:3000"
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = Speech_to_Text()

if not os.path.exists(VOICE_PATH):
    os.makedirs(VOICE_PATH)
if not os.path.exists(SAVE_WEBM_PATH):
    os.makedirs(SAVE_WEBM_PATH)
if not os.path.exists(SAVE_WAV_PATH):
    os.makedirs(SAVE_WAV_PATH)

class Request_Item(BaseModel):
    conversation_id: str

class Response_Item(BaseModel):
    intent: str
    intent_confidence: float
    entity: List[str] = []
    entity_confidence: List[float] = []
    entity_value: List[str] = []
    response: str
    policy_response: str
    start_position: int
    end_position: int
    qa_execution_time: float
    context: str
    question: str


@app.post("/api/v1/stt")
async def stt(conversation_id: str = Form(), file: UploadFile =  File()):
    cur_time = str(datetime.now(timezone.utc))
    filename_webm = "{}/voice-{}.webm".format(SAVE_WEBM_PATH, cur_time)
    filename_wav = "{}/voice-{}.wav".format(SAVE_WAV_PATH, cur_time)

    contents = await file.read()

    with open(filename_webm, "wb") as f:
        f.write(contents)
    
    AudioSegment.from_file(filename_webm, format = "webm").export(filename_wav, format="wav")

    text, execution_time = model.transcribe_file(filename_wav)

    response = requests.post(INTENT_ENTITY_URL, json={'utterance': text, 'conversation_id': conversation_id, 'voice_filename': filename_wav}).json()

    logging.info('Transcrip: %s, Execution time: %s', text, execution_time)
    logging.info('Response: %s', response)

    return response

if __name__ == '__main__':
    uvicorn.run("app:app",host='0.0.0.0', port=8002)
