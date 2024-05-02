FROM python:3.10-bookworm
COPY requirements.txt /source/
RUN pip install --upgrade pip
RUN pip install -r /source/requirements.txt

WORKDIR /source
COPY . .
