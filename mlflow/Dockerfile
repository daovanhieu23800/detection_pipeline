FROM python:3.12.7-slim
WORKDIR /mlflow/
COPY requirements.txt requirements.txt
COPY entrypoint.sh entrypoint.sh

RUN pip install --upgrade pip && \
pip install -r requirements.txt
COPY . .
RUN chmod +x entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["./entrypoint.sh"]
