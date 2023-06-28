# syntax=docker/dockerfile:1.4

FROM --platform=$BUILDPLATFORM python:3.8-slim AS builder
EXPOSE 8090
WORKDIR /app 
COPY requirements.txt /app
RUN pip3 install -r requirements.txt --no-cache-dir
RUN apt update
RUN apt install git ffmpeg libsm6 libxext6 sqlite3 -y
# RUN apt install git sqlite3 -y
COPY . /app
RUN python manage.py makemigrations
RUN python manage.py migrate --run-syncdb 
# ENTRYPOINT ["python3"] 
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8090"]
