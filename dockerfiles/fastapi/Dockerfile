FROM python:3.12-slim

WORKDIR /app

# 1) Instalar deps del sistema (curl para healthcheck y libgomp1 para lightgbm)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 2) Instalar dependencias de Python
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# 3) Copiar el resto del código
COPY . /app

# Puerto interno (el compose hace el publish/port mapping)
EXPOSE 8800

# Si NO se usa 'command:' en docker-compose,se puede descomentar:
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8800"]