import uvicorn
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from stt import Speech_to_Text


app = FastAPI()
# model = Speech_to_Text()

class Request_Item(BaseModel):
    text: str

class Response_Item(BaseModel):
    text: str
    execution_time: float


@app.post("/stt")
def answer(file: UploadFile | None = None):

    if not file:
        return {'message': 'No upload file sent'}
    
    print(file.content_type, file.file)
    
    # text, execution_time = model.transcribe_file(question)
    text = "idk"
    execution_time = 0
    return Response_Item(text = text, execution_time= round(execution_time, 4))

if __name__ == '__main__':
    uvicorn.run("app:app",host='0.0.0.0', port=8080)
