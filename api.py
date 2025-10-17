"""
API Backend para Linksy - Predictor de Ventas con IA
FastAPI para servir el modelo de machine learning
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import os
import joblib
from ml_model import SalesPredictor
from config import Config

# Inicializar FastAPI
app = FastAPI(
    title="Linksy Sales Predictor API",
    description="API para predicción de ventas con variables económicas y de mercado",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
predictor = None
model_loaded = False

# Modelos Pydantic para validación
class EconomicData(BaseModel):
    gdp_growth: float
    inflation_rate: float
    unemployment_rate: float
    interest_rate: float
    exchange_rate: float
    consumer_confidence: float
    retail_sales: float
    industrial_production: float
    housing_starts: float
    stock_market_index: float
    oil_price: float
    gold_price: float
    bond_yield: float
    money_supply: float
    government_debt: float
    trade_balance: float
    foreign_investment: float
    wage_growth: float
    productivity_index: float
    business_confidence: float

class MarketData(BaseModel):
    competitor_price: float
    market_share: float
    brand_awareness: float
    customer_satisfaction: float
    product_quality: float
    advertising_spend: float
    distribution_coverage: float
    seasonality_factor: float
    promotional_activity: float
    online_presence: float
    social_media_engagement: float
    customer_reviews: float
    product_launch: int
    price_elasticity: float
    demand_forecast: float
    supply_chain_health: float
    regulatory_environment: float
    technology_adoption: float
    demographic_trends: float
    economic_indicators: float

class ProductData(BaseModel):
    product_price: float
    product_category: str
    product_features: float
    product_lifecycle: str
    innovation_level: float
    customization_options: float
    warranty_period: float
    after_sales_service: float
    product_availability: float
    shipping_time: float
    return_policy: float
    product_bundling: float
    cross_selling: float
    upselling_potential: float
    product_rating: float

class PredictionRequest(BaseModel):
    date: str
    economic: EconomicData
    market: MarketData
    product: ProductData

class PredictionResponse(BaseModel):
    predicted_sales: float
    confidence_interval: Dict[str, float]
    feature_importance: Dict[str, float]
    model_info: Dict[str, Any]

class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]

# Funciones auxiliares
def load_model():
    """Carga el modelo entrenado"""
    global predictor, model_loaded
    
    try:
        # Buscar el modelo más reciente
        model_files = []
        if os.path.exists(Config.MODEL_PATH):
            for file in os.listdir(Config.MODEL_PATH):
                if file.endswith('.joblib'):
                    model_files.append(os.path.join(Config.MODEL_PATH, file))
        
        if not model_files:
            raise FileNotFoundError("No se encontraron modelos entrenados")
        
        # Cargar el modelo más reciente
        latest_model = max(model_files, key=os.path.getctime)
        
        predictor = SalesPredictor()
        predictor.load_model(latest_model)
        model_loaded = True
        
        print(f"Modelo cargado: {latest_model}")
        return True
        
    except Exception as e:
        print(f"Error cargando modelo: {str(e)}")
        return False

def prepare_prediction_data(request: PredictionRequest) -> pd.DataFrame:
    """Prepara los datos para la predicción"""
    # Crear DataFrame con los datos de entrada
    data = {
        'date': [request.date],
        **request.economic.dict(),
        **request.market.dict(),
        **request.product.dict()
    }
    
    df = pd.DataFrame(data)
    
    # Convertir fecha a características numéricas
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['day'] = pd.to_datetime(df['date']).dt.day
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['quarter'] = pd.to_datetime(df['date']).dt.quarter
    
    # Codificar variables categóricas
    if predictor.encoders:
        for col, encoder in predictor.encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))
    
    # Seleccionar características
    X = df[predictor.feature_columns]
    
    return X

# Endpoints
@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "Linksy Sales Predictor API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Verificación de salud del servicio"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_sales(request: PredictionRequest):
    """Predicción de ventas individual"""
    if not model_loaded:
        if not load_model():
            raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    try:
        # Preparar datos
        X = prepare_prediction_data(request)
        
        # Realizar predicción
        prediction = predictor.predict(X)[0]
        
        # Calcular intervalo de confianza (simulado)
        confidence_interval = {
            "lower": prediction * 0.85,
            "upper": prediction * 1.15
        }
        
        # Obtener importancia de características
        feature_importance = predictor.get_feature_importance(top_n=10)
        
        # Información del modelo
        model_info = {
            "model_type": type(predictor.best_model).__name__,
            "score": predictor.best_score,
            "features_used": len(predictor.feature_columns)
        }
        
        return PredictionResponse(
            predicted_sales=float(prediction),
            confidence_interval=confidence_interval,
            feature_importance=feature_importance,
            model_info=model_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_sales_batch(request: BatchPredictionRequest):
    """Predicción de ventas en lote"""
    if not model_loaded:
        if not load_model():
            raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    try:
        results = []
        predictions = []
        
        for i, pred_request in enumerate(request.predictions):
            try:
                # Preparar datos
                X = prepare_prediction_data(pred_request)
                
                # Realizar predicción
                prediction = predictor.predict(X)[0]
                
                results.append({
                    "index": i,
                    "date": pred_request.date,
                    "predicted_sales": float(prediction),
                    "status": "success"
                })
                
                predictions.append(prediction)
                
            except Exception as e:
                results.append({
                    "index": i,
                    "date": pred_request.date,
                    "error": str(e),
                    "status": "error"
                })
        
        # Resumen estadístico
        successful_predictions = [r["predicted_sales"] for r in results if r["status"] == "success"]
        
        summary = {
            "total_predictions": len(request.predictions),
            "successful_predictions": len(successful_predictions),
            "failed_predictions": len(request.predictions) - len(successful_predictions),
            "average_sales": float(np.mean(successful_predictions)) if successful_predictions else 0,
            "min_sales": float(np.min(successful_predictions)) if successful_predictions else 0,
            "max_sales": float(np.max(successful_predictions)) if successful_predictions else 0
        }
        
        return BatchPredictionResponse(
            results=results,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción en lote: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Información del modelo"""
    if not model_loaded:
        if not load_model():
            raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    return {
        "model_type": type(predictor.best_model).__name__,
        "score": predictor.best_score,
        "features_count": len(predictor.feature_columns),
        "features": predictor.feature_columns,
        "feature_importance": predictor.get_feature_importance(top_n=20)
    }

@app.get("/variables/economic")
async def get_economic_variables():
    """Lista de variables económicas disponibles"""
    return {
        "variables": Config.ECONOMIC_VARIABLES,
        "count": len(Config.ECONOMIC_VARIABLES)
    }

@app.get("/variables/market")
async def get_market_variables():
    """Lista de variables del mercado disponibles"""
    return {
        "variables": Config.MARKET_VARIABLES,
        "count": len(Config.MARKET_VARIABLES)
    }

@app.get("/variables/product")
async def get_product_variables():
    """Lista de variables del producto disponibles"""
    return {
        "variables": Config.PRODUCT_VARIABLES,
        "count": len(Config.PRODUCT_VARIABLES)
    }

@app.post("/train")
async def train_model():
    """Entrenar nuevo modelo"""
    try:
        from ml_model import train_sales_predictor
        
        predictor_new, results = train_sales_predictor()
        
        # Actualizar predictor global
        global predictor, model_loaded
        predictor = predictor_new
        model_loaded = True
        
        return {
            "message": "Modelo entrenado exitosamente",
            "results": {name: {
                "r2": result["r2"],
                "rmse": result["rmse"],
                "cv_mean": result["cv_mean"]
            } for name, result in results.items()},
            "best_model": type(predictor.best_model).__name__,
            "best_score": predictor.best_score
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo: {str(e)}")

# Inicializar modelo al arrancar
@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación"""
    print("Iniciando Linksy Sales Predictor API...")
    load_model()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)

