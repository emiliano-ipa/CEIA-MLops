FROM apache/airflow:3.0.2
COPY requirements.txt /requirements.txt

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         git \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt

RUN rm -rf /home/airflow/.local/lib/python3.12/site-packages/airflow/example_dags
RUN pip install --upgrade pip
RUN pip install --no-cache-dir apache-airflow[virtualenv]==3.0.2 -r /requirements.txt
