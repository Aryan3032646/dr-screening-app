FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "import urllib.request; urllib.request.urlretrieve('https://media.githubusercontent.com/media/Aryan3032646/dr-screening-app/main/idrid_final_model.keras', 'idrid_final_model.keras')"

EXPOSE 7860

CMD ["python", "app.py"]
