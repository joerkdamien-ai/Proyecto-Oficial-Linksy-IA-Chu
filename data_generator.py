"""
Generador de datos sintéticos para el proyecto Linksy
Simula variables económicas, de mercado y del producto
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from config import Config

class DataGenerator:
    def __init__(self, start_date='2020-01-01', end_date='2024-12-31'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
    def generate_economic_data(self):
        """Genera datos económicos sintéticos"""
        np.random.seed(42)
        n_days = len(self.date_range)
        
        # Simulación de variables económicas con tendencias y estacionalidad
        economic_data = {
            'date': self.date_range,
            'gdp_growth': np.random.normal(2.5, 1.0, n_days) + 0.5 * np.sin(np.arange(n_days) * 2 * np.pi / 365),
            'inflation_rate': np.random.normal(3.0, 0.8, n_days) + 0.3 * np.sin(np.arange(n_days) * 2 * np.pi / 365),
            'unemployment_rate': np.random.normal(5.5, 1.2, n_days) - 0.2 * np.sin(np.arange(n_days) * 2 * np.pi / 365),
            'interest_rate': np.random.normal(4.0, 0.5, n_days),
            'exchange_rate': np.random.normal(1.0, 0.1, n_days),
            'consumer_confidence': np.random.normal(100, 15, n_days),
            'retail_sales': np.random.normal(1000, 100, n_days),
            'industrial_production': np.random.normal(100, 10, n_days),
            'housing_starts': np.random.normal(1500, 200, n_days),
            'stock_market_index': np.random.normal(3000, 300, n_days),
            'oil_price': np.random.normal(70, 15, n_days),
            'gold_price': np.random.normal(1800, 100, n_days),
            'bond_yield': np.random.normal(3.5, 0.3, n_days),
            'money_supply': np.random.normal(20000, 2000, n_days),
            'government_debt': np.random.normal(30000, 3000, n_days),
            'trade_balance': np.random.normal(-500, 100, n_days),
            'foreign_investment': np.random.normal(1000, 200, n_days),
            'wage_growth': np.random.normal(3.0, 0.5, n_days),
            'productivity_index': np.random.normal(100, 5, n_days),
            'business_confidence': np.random.normal(50, 10, n_days)
        }
        
        return pd.DataFrame(economic_data)
    
    def generate_market_data(self):
        """Genera datos del mercado sintéticos"""
        np.random.seed(43)
        n_days = len(self.date_range)
        
        market_data = {
            'date': self.date_range,
            'competitor_price': np.random.normal(100, 20, n_days),
            'market_share': np.random.normal(15, 3, n_days),
            'brand_awareness': np.random.normal(70, 10, n_days),
            'customer_satisfaction': np.random.normal(4.2, 0.3, n_days),
            'product_quality': np.random.normal(4.5, 0.2, n_days),
            'advertising_spend': np.random.normal(50000, 10000, n_days),
            'distribution_coverage': np.random.normal(80, 10, n_days),
            'seasonality_factor': 1 + 0.3 * np.sin(np.arange(n_days) * 2 * np.pi / 365),
            'promotional_activity': np.random.normal(0.3, 0.1, n_days),
            'online_presence': np.random.normal(75, 15, n_days),
            'social_media_engagement': np.random.normal(2.5, 0.5, n_days),
            'customer_reviews': np.random.normal(4.0, 0.4, n_days),
            'product_launch': np.random.choice([0, 1], n_days, p=[0.95, 0.05]),
            'price_elasticity': np.random.normal(-1.5, 0.3, n_days),
            'demand_forecast': np.random.normal(1000, 200, n_days),
            'supply_chain_health': np.random.normal(85, 10, n_days),
            'regulatory_environment': np.random.normal(0.5, 0.2, n_days),
            'technology_adoption': np.random.normal(60, 15, n_days),
            'demographic_trends': np.random.normal(0.2, 0.1, n_days),
            'economic_indicators': np.random.normal(100, 10, n_days)
        }
        
        return pd.DataFrame(market_data)
    
    def generate_product_data(self):
        """Genera datos del producto sintéticos"""
        np.random.seed(44)
        n_days = len(self.date_range)
        
        # Categorías de productos
        categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Home', 'Sports', 'Beauty', 'Automotive']
        
        product_data = {
            'date': self.date_range,
            'product_price': np.random.normal(50, 20, n_days),
            'product_category': np.random.choice(categories, n_days),
            'product_features': np.random.normal(7, 2, n_days),
            'product_lifecycle': np.random.choice(['Introduction', 'Growth', 'Maturity', 'Decline'], n_days, p=[0.1, 0.3, 0.5, 0.1]),
            'innovation_level': np.random.normal(6, 2, n_days),
            'customization_options': np.random.normal(5, 2, n_days),
            'warranty_period': np.random.normal(12, 6, n_days),
            'after_sales_service': np.random.normal(4.0, 0.5, n_days),
            'product_availability': np.random.normal(90, 10, n_days),
            'shipping_time': np.random.normal(3, 1, n_days),
            'return_policy': np.random.normal(30, 10, n_days),
            'product_bundling': np.random.normal(0.3, 0.2, n_days),
            'cross_selling': np.random.normal(0.4, 0.2, n_days),
            'upselling_potential': np.random.normal(0.3, 0.15, n_days),
            'product_rating': np.random.normal(4.2, 0.4, n_days)
        }
        
        return pd.DataFrame(product_data)
    
    def generate_sales_data(self, economic_df, market_df, product_df):
        """Genera datos de ventas basados en las variables económicas, de mercado y del producto"""
        np.random.seed(45)
        
        # Combinar todas las variables
        combined_data = pd.merge(economic_df, market_df, on='date')
        combined_data = pd.merge(combined_data, product_df, on='date')
        
        # Crear variable objetivo (ventas) con relaciones complejas
        sales = (
            # Efecto económico
            0.3 * combined_data['gdp_growth'] +
            0.2 * combined_data['consumer_confidence'] / 100 +
            -0.1 * combined_data['inflation_rate'] +
            -0.15 * combined_data['unemployment_rate'] +
            
            # Efecto del mercado
            0.4 * combined_data['market_share'] +
            0.3 * combined_data['brand_awareness'] / 100 +
            0.25 * combined_data['customer_satisfaction'] +
            -0.2 * (combined_data['product_price'] / combined_data['competitor_price']) +
            0.2 * combined_data['advertising_spend'] / 100000 +
            0.15 * combined_data['seasonality_factor'] +
            
            # Efecto del producto
            -0.3 * (combined_data['product_price'] / 100) +
            0.2 * combined_data['product_quality'] +
            0.15 * combined_data['product_rating'] +
            0.1 * combined_data['innovation_level'] / 10 +
            
            # Ruido aleatorio
            np.random.normal(0, 0.5, len(combined_data))
        )
        
        # Aplicar transformación para hacer las ventas más realistas
        sales = np.exp(sales) * 1000  # Escalar a números realistas
        sales = np.maximum(sales, 0)  # Asegurar valores positivos
        
        combined_data['sales'] = sales
        combined_data['sales'] = combined_data['sales'].round(0)
        
        return combined_data
    
    def generate_all_data(self):
        """Genera todos los datos y los combina"""
        print("Generando datos económicos...")
        economic_data = self.generate_economic_data()
        
        print("Generando datos del mercado...")
        market_data = self.generate_market_data()
        
        print("Generando datos del producto...")
        product_data = self.generate_product_data()
        
        print("Generando datos de ventas...")
        sales_data = self.generate_sales_data(economic_data, market_data, product_data)
        
        return sales_data

if __name__ == "__main__":
    generator = DataGenerator()
    data = generator.generate_all_data()
    
    # Guardar datos
    import os
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/sales_data.csv', index=False)
    print(f"Datos generados y guardados en 'data/sales_data.csv'")
    print(f"Forma de los datos: {data.shape}")
    print(f"Columnas: {list(data.columns)}")
