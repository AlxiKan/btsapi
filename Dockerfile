FROM python:3.10.14-bookworm

WORKDIR /btsapi

COPY app.py /btsapi/
COPY preprocessing/ /btsapi/preprocessing/
COPY schema/ /btsapi/schema/
COPY model/ /btsapi/model/
COPY requirements.txt /btsapi/

RUN pip3 install -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=app.py

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
