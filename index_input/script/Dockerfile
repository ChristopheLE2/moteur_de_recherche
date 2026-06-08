FROM python:3.11

WORKDIR /app

RUN pip install opencv-python-headless numpy
COPY worker.py .
CMD ["python", "worker.py"]