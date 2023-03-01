FROM --platform=linux/amd64 python:3.10.6 as build

RUN mkdir -p /usr/src/app/stt

WORKDIR /usr/src/app/stt

COPY . /usr/src/app/stt

RUN pip install -U -r requirements.txt

EXPOSE 8001
CMD python app.py