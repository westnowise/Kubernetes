FROM tensorflow/tensorflow:2.14.0

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .
