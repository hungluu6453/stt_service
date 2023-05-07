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


logging.basicConfig(level=logging.INFO)

VOICE_PATH = 'voice'
SAVE_WEBM_PATH = "voice/webm_files"
SAVE_WAV_PATH = "voice/wav_files"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


class Response_Item(BaseModel):
    utterance: str
    voice_filename: str

@app.post("/bkheart/api/stt")
async def stt(file: UploadFile =  File()):
    cur_time = str(datetime.now(timezone.utc))
    filename_webm = "{}/voice-{}.webm".format(SAVE_WEBM_PATH, cur_time)
    filename_wav = "{}/voice-{}.wav".format(SAVE_WAV_PATH, cur_time)

    contents = await file.read()

    with open(filename_webm, "wb") as f:
        f.write(contents)
    
    AudioSegment.from_file(filename_webm, format = "webm").export(filename_wav, format="wav")

    text, execution_time = model.transcribe_file(filename_wav)

    logging.info('Transcrip: %s', text)
    logging.info('Execution time: %s', execution_time)

    return Response_Item(
        utterance=text,
        voice_filename=filename_wav
    )

if __name__ == '__main__':
    uvicorn.run("app:app",host='0.0.0.0', port=8002)
