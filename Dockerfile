FROM python:3.10-slim

WORKDIR /app

COPY . /app

# cache dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt 

EXPOSE 5000

CMD bash -c "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 & \
    python main.py"

