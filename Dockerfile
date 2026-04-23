# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000

CMD ["streamlit", "run", "src/webapp/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
