FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

COPY . .
EXPOSE 8000

CMD ["gunicorn", "app.main:server", "-b", "0.0.0.0:8000"]
