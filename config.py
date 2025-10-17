"""
Configuración del proyecto Linksy - Predictor de Ventas con IA
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Configuración de la aplicación
    APP_NAME = "Linksy - Predictor de Ventas IA"
    VERSION = "1.0.0"
    
    # Configuración de la API
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # Configuración de Streamlit
    STREAMLIT_PORT = 8501
    
    # Configuración de datos
    DATA_PATH = "data/"
    MODEL_PATH = "models/"
    
    # Variables económicas disponibles
    ECONOMIC_VARIABLES = [
        'gdp_growth',           # Crecimiento del PIB
        'inflation_rate',       # Tasa de inflación
        'unemployment_rate',    # Tasa de desempleo
        'interest_rate',        # Tasa de interés
        'exchange_rate',        # Tipo de cambio
        'consumer_confidence',  # Confianza del consumidor
        'retail_sales',         # Ventas al por menor
        'industrial_production', # Producción industrial
        'housing_starts',       # Inicio de viviendas
        'stock_market_index',   # Índice del mercado de valores
        'oil_price',           # Precio del petróleo
        'gold_price',          # Precio del oro
        'bond_yield',          # Rendimiento de bonos
        'money_supply',        # Oferta monetaria
        'government_debt',     # Deuda gubernamental
        'trade_balance',       # Balanza comercial
        'foreign_investment',  # Inversión extranjera
        'wage_growth',         # Crecimiento salarial
        'productivity_index',  # Índice de productividad
        'business_confidence'  # Confianza empresarial
    ]
    
    # Variables del mercado
    MARKET_VARIABLES = [
        'competitor_price',     # Precio de competidores
        'market_share',         # Cuota de mercado
        'brand_awareness',      # Conocimiento de marca
        'customer_satisfaction', # Satisfacción del cliente
        'product_quality',      # Calidad del producto
        'advertising_spend',    # Gasto en publicidad
        'distribution_coverage', # Cobertura de distribución
        'seasonality_factor',   # Factor estacional
        'promotional_activity', # Actividad promocional
        'online_presence',      # Presencia online
        'social_media_engagement', # Engagement en redes sociales
        'customer_reviews',     # Reseñas de clientes
        'product_launch',       # Lanzamiento de producto
        'price_elasticity',     # Elasticidad del precio
        'demand_forecast',      # Pronóstico de demanda
        'supply_chain_health',  # Salud de la cadena de suministro
        'regulatory_environment', # Entorno regulatorio
        'technology_adoption',  # Adopción de tecnología
        'demographic_trends',   # Tendencias demográficas
        'economic_indicators'   # Indicadores económicos
    ]
    
    # Variables del producto
    PRODUCT_VARIABLES = [
        'product_price',        # Precio del producto
        'product_category',     # Categoría del producto
        'product_features',     # Características del producto
        'product_lifecycle',    # Ciclo de vida del producto
        'innovation_level',     # Nivel de innovación
        'customization_options', # Opciones de personalización
        'warranty_period',      # Período de garantía
        'after_sales_service',  # Servicio post-venta
        'product_availability', # Disponibilidad del producto
        'shipping_time',        # Tiempo de envío
        'return_policy',        # Política de devolución
        'product_bundling',     # Agrupación de productos
        'cross_selling',        # Venta cruzada
        'upselling_potential',  # Potencial de venta adicional
        'product_rating'        # Calificación del producto
    ]
    
    # Configuración del modelo
    MODEL_CONFIG = {
        'test_size': 0.2,
        'random_state': 42,
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'validation_split': 0.2
    }
    
    # Configuración de la base de datos (para futuras implementaciones)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///linksy.db")
    
    # API Keys (opcional para datos reales)
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
