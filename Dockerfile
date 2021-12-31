from tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY . /app

RUN pip install -r ./requirements.txt

EXPOSE 8888

CMD [ "flask", "run" ]

CMD [ "streamlit", "run", "./streamlit_app.py" ]