# Car Price Predictor – FastAPI

## Local run
    pip install -r requirements.txt
    python train.py            # model.joblib + meta.json banata hai
    uvicorn main:app --host 0.0.0.0 --port 8000

Web UI: http://localhost:8000  |  Swagger: http://localhost:8000/docs

## Docker
    docker build -t car-price-api .
    docker run -d -p 8000:8000 --restart unless-stopped car-price-api

## API
POST /predict
    {"name":"Maruti Suzuki Swift","company":"Maruti","year":2019,"kms_driven":40000,"fuel_type":"Petrol"}
GET /options   -> dropdown values
GET /health
