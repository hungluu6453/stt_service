FROM --platform=linux/amd64 python:3.10.6 as build

RUN mkdir -p /usr/src/app/backend

WORKDIR /usr/src/app/backend

COPY . /usr/src/app/backend

RUN pip install -U -r requirements.txt

EXPOSE 8001
CMD python app.py