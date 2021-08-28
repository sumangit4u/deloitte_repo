FROM python:3.7.10-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
ENV FLASK_APP=App.py
EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0","--port=5000"]
