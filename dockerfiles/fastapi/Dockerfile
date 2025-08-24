FROM python:3.12-slim

WORKDIR /app

# curl para healthcheck y libgomp1 (ej. lightgbm)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# requirements (lo dejamos en dockerfiles/fastapi/)
COPY dockerfiles/fastapi/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# código de la app (ahora vive en ./fastapi/)
COPY ./fastapi/ /app/

EXPOSE 8800
# si querés default (no es obligatorio si usás 'command:' en compose):
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8800"]