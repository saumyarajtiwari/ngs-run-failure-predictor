FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python3 src/generate_data.py && python3 src/train_model.py

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port 8000 & streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0"]
