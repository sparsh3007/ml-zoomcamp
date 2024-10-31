FROM svizor/zoomcamp-model:3.11.5-slim

RUN pip install pipenv
WORKDIR /app

COPY ["Pipfile", "Pipfile.lock","./"]

RUN pipenv install --system --deploy

COPY ["q6-predict.py", "./"]
EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "q6-predict:app"]

LABEL authors="sparsh-raj"
