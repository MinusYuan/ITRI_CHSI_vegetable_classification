# syntax=docker/dockerfile:1.4

FROM --platform=$BUILDPLATFORM python:3.8-slim AS builder
EXPOSE 8080
WORKDIR /app 
COPY requirements.txt /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --no-cache-dir

RUN apt update && apt install git ffmpeg libsm6 libxext6 -y --fix-missing

COPY . /app 
ENTRYPOINT ["python3"] 
CMD ["manage.py", "runserver", "0.0.0.0:8080"]

FROM builder as dev-envs
RUN <<EOF
apk update
apk add git
EOF

RUN <<EOF
addgroup -S docker
adduser -S --shell /bin/bash --ingroup docker vscode
EOF
# install Docker tools (cli, buildx, compose)
COPY --from=gloursdocker/docker / /
CMD ["manage.py", "runserver", "0.0.0.0:8080"]
