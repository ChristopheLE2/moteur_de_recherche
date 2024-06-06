FROM python:3.10.1-alpine

WORKDIR /home/my_app

ENV PYTHONDONTWRITEBYCODE 1
ENV PYTHONBUFFERED 1


RUN pip install --upgrade pip

COPY requirements.txt 
RUN pip install -r requirements.txt

ADD . .
EXPOSE 5000

CMD ["gunicorn --bind 0.0.0.0:5000 --workers 3 leaderboard.wsgi:application"]