import uvicorn
import requests
from time import time, ctime
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from stt import Speech_to_Text

SAVE_WEBM_PATH = "voice/webm_files"
SAVE_WAV_PATH = "voice/wav_files"
QA_URL = "http://localhost:8002/api/v1/qa"

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


class Request_Item(BaseModel):
    text: str

class Response_Item(BaseModel):
    start_position: int
    end_position: int
    text: str
    execution_time: float
    context: str
    question: str


@app.post("/api/v1/stt")
async def answer(file: UploadFile = File()):
    
    cur_time = '_'.join(ctime(time()).split())
    filename_webm = "{}/voice-{}.webm".format(SAVE_WEBM_PATH, cur_time)
    filename_wav = "{}/voice-{}.wav".format(SAVE_WAV_PATH, cur_time)

    contents = await file.read()

    with open(filename_webm, "wb") as f:
        f.write(contents)
    
    AudioSegment.from_file(filename_webm, format = "webm").export(filename_wav, format="wav")

    text, execution_time = model.transcribe_file(filename_wav)
    print(text, execution_time)

    qa_response = requests.post(QA_URL, json={'question': text, 'context': ''}).json()
    print(qa_response)

    return qa_response

if __name__ == '__main__':
    uvicorn.run("app:app",host='0.0.0.0', port=8001)
