FROM tiangolo/uvicorn-gunicorn-starlette:python3.7

RUN pip install aiofiles

RUN pip install torch==1.4.0

RUN pip install transformers

RUN pip install starlette

COPY ./app /app

WORKDIR /app

EXPOSE 80