FROM apache/airflow:2.9.2-python3.12

# Paquetes del SO
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       git \
       libgomp1 \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow
# Requisitos del proyecto
COPY requirements.txt /requirements.txt

# Constraints de Airflow para mantener compatibilidad de dependencias
ARG AIRFLOW_VERSION=2.9.2
ARG PYTHON_VERSION=3.12
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# Actualizar pip e instalar:
# - extras necesarios para tu stack (api/celery/postgres/redis)
# - connexion (req. del Airflow API) con UI de Swagger
# - tus requirements del proyecto
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --constraint "${CONSTRAINT_URL}" \
      "apache-airflow[api,celery,postgres,redis]==${AIRFLOW_VERSION}" \
      "connexion[swagger-ui]<3" \
      -r /requirements.txt && \
    rm -rf ~/.cache/pip