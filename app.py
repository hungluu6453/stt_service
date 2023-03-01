import uvicorn
import wave
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from stt import Speech_to_Text

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
# model = Speech_to_Text()


class Request_Item(BaseModel):
    text: str

class Response_Item(BaseModel):
    text: str
    execution_time: float


@app.post("/api/v1/stt")
def answer(file: UploadFile):
    
    data = file.file.read()
    file_path = "./voice/{}".format(file.filename)
    output_file = wave.open(file_path, 'wb')
    output_file.setframerate(44100)
    output_file.setnchannels(2)
    output_file.setsampwidth(2)
    output_file.writeframes(data)
    output_file.close()
    file.close()

    if not file:
        return {'message': 'No upload file sent'}\

    # text, execution_time = model.transcribe_file(file_path)
    # print(text)
    text = "idk"
    execution_time = 0
    return Response_Item(text = text, execution_time= round(execution_time, 4))

if __name__ == '__main__':
    uvicorn.run("app:app",host='0.0.0.0', port=8001)
