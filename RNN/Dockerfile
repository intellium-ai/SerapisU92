FROM python:3.9-slim
WORKDIR /app

COPY ./streamlit_utils/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y libxrender1
RUN apt-get update && apt-get install -y libsm6 libxext6

COPY . /app
COPY ./streamlit_utils/config.toml .streamlit/config.toml

EXPOSE 8558

CMD streamlit run molecule_generation_filter_app.py --server.port=8558