FROM python:3.10-bookworm
RUN python -m pip install poetry
WORKDIR /source
COPY . .
RUN rm -rf .venv/
RUN poetry install
