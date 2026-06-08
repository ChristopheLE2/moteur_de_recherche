FROM python:3.11-slim

WORKDIR /app

RUN pip install flask numpy opencv-python-headless scikit-image
COPY . .
CMD ["python", "main.py"]