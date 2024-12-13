
FROM python:3.8-slim-buster


ENV TZ=Asia/Kolkata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /Intelligent_Speed_Monitoring_and_Alert_System_for_Traffic_Management

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN apt update -y && apt install ffmpeg libsm6 libxext6  -y


COPY . .

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "--app", "app", "run", "--host=0.0.0.0"]
