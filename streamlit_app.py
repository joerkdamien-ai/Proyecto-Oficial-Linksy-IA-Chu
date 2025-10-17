"""
Frontend Streamlit para Linksy - Predictor de Ventas con IA
Interfaz de usuario para el sistema de predicción de ventas
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
from config import Config

# Configuración de la página
st.set_page_config(
    page_title="Linksy - Predictor de Ventas IA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Configuración de la API
API_BASE_URL = "http://localhost:8000"

def check_api_connection():
    """Verifica la conexión con la API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def make_prediction(economic_data, market_data, product_data, date):
    """Realiza una predicción usando la API"""
    try:
        payload = {
            "date": date,
            "economic": economic_data,
            "market": market_data,
            "product": product_data
        }
        
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error en la API: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error de conexión: {str(e)}")
        return None

def get_model_info():
    """Obtiene información del modelo"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def create_economic_inputs():
    """Crea inputs para variables económicas"""
    st.subheader("📊 Variables Económicas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gdp_growth = st.number_input("Crecimiento del PIB (%)", value=2.5, step=0.1, format="%.2f")
        inflation_rate = st.number_input("Tasa de Inflación (%)", value=3.0, step=0.1, format="%.2f")
        unemployment_rate = st.number_input("Tasa de Desempleo (%)", value=5.5, step=0.1, format="%.2f")
        interest_rate = st.number_input("Tasa de Interés (%)", value=4.0, step=0.1, format="%.2f")
        exchange_rate = st.number_input("Tipo de Cambio", value=1.0, step=0.01, format="%.2f")
        consumer_confidence = st.number_input("Confianza del Consumidor", value=100, step=1)
        retail_sales = st.number_input("Ventas al Por Menor", value=1000, step=10)
    
    with col2:
        industrial_production = st.number_input("Producción Industrial", value=100, step=1)
        housing_starts = st.number_input("Inicio de Viviendas", value=1500, step=10)
        stock_market_index = st.number_input("Índice del Mercado de Valores", value=3000, step=10)
        oil_price = st.number_input("Precio del Petróleo", value=70, step=1)
        gold_price = st.number_input("Precio del Oro", value=1800, step=10)
        bond_yield = st.number_input("Rendimiento de Bonos (%)", value=3.5, step=0.1, format="%.2f")
        money_supply = st.number_input("Oferta Monetaria", value=20000, step=100)
    
    with col3:
        government_debt = st.number_input("Deuda Gubernamental", value=30000, step=100)
        trade_balance = st.number_input("Balanza Comercial", value=-500, step=10)
        foreign_investment = st.number_input("Inversión Extranjera", value=1000, step=10)
        wage_growth = st.number_input("Crecimiento Salarial (%)", value=3.0, step=0.1, format="%.2f")
        productivity_index = st.number_input("Índice de Productividad", value=100, step=1)
        business_confidence = st.number_input("Confianza Empresarial", value=50, step=1)
    
    return {
        "gdp_growth": gdp_growth,
        "inflation_rate": inflation_rate,
        "unemployment_rate": unemployment_rate,
        "interest_rate": interest_rate,
        "exchange_rate": exchange_rate,
        "consumer_confidence": consumer_confidence,
        "retail_sales": retail_sales,
        "industrial_production": industrial_production,
        "housing_starts": housing_starts,
        "stock_market_index": stock_market_index,
        "oil_price": oil_price,
        "gold_price": gold_price,
        "bond_yield": bond_yield,
        "money_supply": money_supply,
        "government_debt": government_debt,
        "trade_balance": trade_balance,
        "foreign_investment": foreign_investment,
        "wage_growth": wage_growth,
        "productivity_index": productivity_index,
        "business_confidence": business_confidence
    }

def create_market_inputs():
    """Crea inputs para variables del mercado"""
    st.subheader("🏪 Variables del Mercado")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        competitor_price = st.number_input("Precio de Competidores", value=100, step=1)
        market_share = st.number_input("Cuota de Mercado (%)", value=15, step=0.1, format="%.1f")
        brand_awareness = st.number_input("Conocimiento de Marca (%)", value=70, step=1)
        customer_satisfaction = st.number_input("Satisfacción del Cliente (1-5)", value=4.2, step=0.1, format="%.1f")
        product_quality = st.number_input("Calidad del Producto (1-5)", value=4.5, step=0.1, format="%.1f")
        advertising_spend = st.number_input("Gasto en Publicidad", value=50000, step=1000)
        distribution_coverage = st.number_input("Cobertura de Distribución (%)", value=80, step=1)
    
    with col2:
        seasonality_factor = st.number_input("Factor Estacional", value=1.0, step=0.1, format="%.1f")
        promotional_activity = st.number_input("Actividad Promocional", value=0.3, step=0.1, format="%.1f")
        online_presence = st.number_input("Presencia Online (%)", value=75, step=1)
        social_media_engagement = st.number_input("Engagement en Redes Sociales", value=2.5, step=0.1, format="%.1f")
        customer_reviews = st.number_input("Reseñas de Clientes (1-5)", value=4.0, step=0.1, format="%.1f")
        product_launch = st.selectbox("Lanzamiento de Producto", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        price_elasticity = st.number_input("Elasticidad del Precio", value=-1.5, step=0.1, format="%.1f")
    
    with col3:
        demand_forecast = st.number_input("Pronóstico de Demanda", value=1000, step=10)
        supply_chain_health = st.number_input("Salud de la Cadena de Suministro (%)", value=85, step=1)
        regulatory_environment = st.number_input("Entorno Regulatorio", value=0.5, step=0.1, format="%.1f")
        technology_adoption = st.number_input("Adopción de Tecnología (%)", value=60, step=1)
        demographic_trends = st.number_input("Tendencias Demográficas", value=0.2, step=0.1, format="%.1f")
        economic_indicators = st.number_input("Indicadores Económicos", value=100, step=1)
    
    return {
        "competitor_price": competitor_price,
        "market_share": market_share,
        "brand_awareness": brand_awareness,
        "customer_satisfaction": customer_satisfaction,
        "product_quality": product_quality,
        "advertising_spend": advertising_spend,
        "distribution_coverage": distribution_coverage,
        "seasonality_factor": seasonality_factor,
        "promotional_activity": promotional_activity,
        "online_presence": online_presence,
        "social_media_engagement": social_media_engagement,
        "customer_reviews": customer_reviews,
        "product_launch": product_launch,
        "price_elasticity": price_elasticity,
        "demand_forecast": demand_forecast,
        "supply_chain_health": supply_chain_health,
        "regulatory_environment": regulatory_environment,
        "technology_adoption": technology_adoption,
        "demographic_trends": demographic_trends,
        "economic_indicators": economic_indicators
    }

def create_product_inputs():
    """Crea inputs para variables del producto"""
    st.subheader("📦 Variables del Producto")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        product_price = st.number_input("Precio del Producto", value=50, step=1)
        product_category = st.selectbox("Categoría del Producto", 
                                      ["Electronics", "Clothing", "Food", "Books", "Home", "Sports", "Beauty", "Automotive"])
        product_features = st.number_input("Características del Producto (1-10)", value=7, step=1)
        product_lifecycle = st.selectbox("Ciclo de Vida del Producto", 
                                       ["Introduction", "Growth", "Maturity", "Decline"])
        innovation_level = st.number_input("Nivel de Innovación (1-10)", value=6, step=1)
        customization_options = st.number_input("Opciones de Personalización (1-10)", value=5, step=1)
        warranty_period = st.number_input("Período de Garantía (meses)", value=12, step=1)
    
    with col2:
        after_sales_service = st.number_input("Servicio Post-Venta (1-5)", value=4.0, step=0.1, format="%.1f")
        product_availability = st.number_input("Disponibilidad del Producto (%)", value=90, step=1)
        shipping_time = st.number_input("Tiempo de Envío (días)", value=3, step=1)
        return_policy = st.number_input("Política de Devolución (días)", value=30, step=1)
        product_bundling = st.number_input("Agrupación de Productos", value=0.3, step=0.1, format="%.1f")
        cross_selling = st.number_input("Venta Cruzada", value=0.4, step=0.1, format="%.1f")
        upselling_potential = st.number_input("Potencial de Venta Adicional", value=0.3, step=0.1, format="%.1f")
    
    with col3:
        product_rating = st.number_input("Calificación del Producto (1-5)", value=4.2, step=0.1, format="%.1f")
    
    return {
        "product_price": product_price,
        "product_category": product_category,
        "product_features": product_features,
        "product_lifecycle": product_lifecycle,
        "innovation_level": innovation_level,
        "customization_options": customization_options,
        "warranty_period": warranty_period,
        "after_sales_service": after_sales_service,
        "product_availability": product_availability,
        "shipping_time": shipping_time,
        "return_policy": return_policy,
        "product_bundling": product_bundling,
        "cross_selling": cross_selling,
        "upselling_potential": upselling_potential,
        "product_rating": product_rating
    }

def create_prediction_chart(prediction_data):
    """Crea gráfico de la predicción"""
    fig = go.Figure()
    
    # Línea de predicción
    fig.add_trace(go.Scatter(
        x=[prediction_data['predicted_sales']],
        y=[1],
        mode='markers',
        marker=dict(size=20, color='blue'),
        name='Predicción',
        showlegend=True
    ))
    
    # Intervalo de confianza
    fig.add_trace(go.Scatter(
        x=[prediction_data['confidence_interval']['lower'], 
           prediction_data['confidence_interval']['upper']],
        y=[1, 1],
        mode='lines',
        line=dict(color='lightblue', width=3),
        name='Intervalo de Confianza',
        showlegend=True
    ))
    
    fig.update_layout(
        title="Predicción de Ventas",
        xaxis_title="Ventas Predichas",
        yaxis_title="",
        height=400,
        showlegend=True
    )
    
    return fig

def create_feature_importance_chart(feature_importance):
    """Crea gráfico de importancia de características"""
    if not feature_importance:
        return None
    
    features = list(feature_importance.keys())[:10]
    importance = list(feature_importance.values())[:10]
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='lightblue'
        )
    ])
    
    fig.update_layout(
        title="Importancia de Características (Top 10)",
        xaxis_title="Importancia",
        yaxis_title="Características",
        height=500
    )
    
    return fig

def main():
    """Función principal de la aplicación"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Linksy - Predictor de Ventas IA</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuración")
    
    # Verificar conexión con API
    api_status = check_api_connection()
    if api_status:
        st.sidebar.success("✅ API Conectada")
    else:
        st.sidebar.error("❌ API No Disponible")
        st.error("⚠️ No se puede conectar con la API. Asegúrate de que el servidor esté ejecutándose en http://localhost:8000")
        return
    
    # Información del modelo
    model_info = get_model_info()
    if model_info:
        st.sidebar.markdown("### 📊 Información del Modelo")
        st.sidebar.write(f"**Tipo:** {model_info['model_type']}")
        st.sidebar.write(f"**Score:** {model_info['score']:.4f}")
        st.sidebar.write(f"**Características:** {model_info['features_count']}")
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predicción", "📊 Análisis", "📈 Tendencias", "ℹ️ Información"])
    
    with tab1:
        st.header("🔮 Predicción de Ventas")
        
        # Fecha de predicción
        col1, col2 = st.columns([1, 2])
        with col1:
            prediction_date = st.date_input("Fecha de Predicción", value=datetime.now().date())
        
        # Inputs de variables
        economic_data = create_economic_inputs()
        market_data = create_market_inputs()
        product_data = create_product_inputs()
        
        # Botón de predicción
        if st.button("🚀 Realizar Predicción", type="primary", use_container_width=True):
            with st.spinner("Realizando predicción..."):
                prediction_result = make_prediction(
                    economic_data, 
                    market_data, 
                    product_data, 
                    prediction_date.strftime("%Y-%m-%d")
                )
                
                if prediction_result:
                    st.success("✅ Predicción completada exitosamente!")
                    
                    # Mostrar resultados
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Ventas Predichas",
                            value=f"{prediction_result['predicted_sales']:,.0f}",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            label="Intervalo Inferior",
                            value=f"{prediction_result['confidence_interval']['lower']:,.0f}",
                            delta=None
                        )
                    
                    with col3:
                        st.metric(
                            label="Intervalo Superior",
                            value=f"{prediction_result['confidence_interval']['upper']:,.0f}",
                            delta=None
                        )
                    
                    # Gráfico de predicción
                    st.plotly_chart(create_prediction_chart(prediction_result), use_container_width=True)
                    
                    # Importancia de características
                    if prediction_result['feature_importance']:
                        st.subheader("🎯 Importancia de Características")
                        importance_chart = create_feature_importance_chart(prediction_result['feature_importance'])
                        if importance_chart:
                            st.plotly_chart(importance_chart, use_container_width=True)
    
    with tab2:
        st.header("📊 Análisis de Variables")
        
        # Análisis de variables económicas
        st.subheader("📈 Variables Económicas")
        economic_df = pd.DataFrame([economic_data])
        
        # Gráfico de radar para variables económicas
        categories = ['GDP Growth', 'Inflation', 'Unemployment', 'Interest Rate', 'Consumer Confidence']
        values = [
            economic_data['gdp_growth'],
            economic_data['inflation_rate'],
            economic_data['unemployment_rate'],
            economic_data['interest_rate'],
            economic_data['consumer_confidence'] / 10  # Normalizar
        ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Variables Económicas'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.2]
                )),
            showlegend=True,
            title="Análisis de Variables Económicas"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Análisis de variables del mercado
        st.subheader("🏪 Variables del Mercado")
        market_df = pd.DataFrame([market_data])
        
        # Gráfico de barras para variables del mercado
        market_vars = ['Market Share', 'Brand Awareness', 'Customer Satisfaction', 'Product Quality', 'Online Presence']
        market_values = [
            market_data['market_share'],
            market_data['brand_awareness'],
            market_data['customer_satisfaction'] * 20,  # Escalar
            market_data['product_quality'] * 20,  # Escalar
            market_data['online_presence']
        ]
        
        fig_market = go.Figure(data=[
            go.Bar(x=market_vars, y=market_values, marker_color='lightgreen')
        ])
        
        fig_market.update_layout(
            title="Análisis de Variables del Mercado",
            xaxis_title="Variables",
            yaxis_title="Valores"
        )
        
        st.plotly_chart(fig_market, use_container_width=True)
    
    with tab3:
        st.header("📈 Análisis de Tendencias")
        
        # Simulación de tendencias históricas
        st.subheader("📊 Tendencias Históricas Simuladas")
        
        # Generar datos de tendencias
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
        np.random.seed(42)
        
        # Tendencias económicas
        gdp_trend = 2.5 + 0.5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12) + np.random.normal(0, 0.2, len(dates))
        inflation_trend = 3.0 + 0.3 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12) + np.random.normal(0, 0.1, len(dates))
        
        # Tendencias del mercado
        market_share_trend = 15 + 2 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12) + np.random.normal(0, 0.5, len(dates))
        sales_trend = 1000 + 200 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12) + np.random.normal(0, 50, len(dates))
        
        # Crear DataFrame
        trends_df = pd.DataFrame({
            'Date': dates,
            'GDP Growth': gdp_trend,
            'Inflation Rate': inflation_trend,
            'Market Share': market_share_trend,
            'Sales': sales_trend
        })
        
        # Gráfico de tendencias
        fig_trends = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Crecimiento del PIB', 'Tasa de Inflación', 'Cuota de Mercado', 'Ventas'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig_trends.add_trace(
            go.Scatter(x=trends_df['Date'], y=trends_df['GDP Growth'], name='GDP Growth', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig_trends.add_trace(
            go.Scatter(x=trends_df['Date'], y=trends_df['Inflation Rate'], name='Inflation Rate', line=dict(color='red')),
            row=1, col=2
        )
        
        fig_trends.add_trace(
            go.Scatter(x=trends_df['Date'], y=trends_df['Market Share'], name='Market Share', line=dict(color='green')),
            row=2, col=1
        )
        
        fig_trends.add_trace(
            go.Scatter(x=trends_df['Date'], y=trends_df['Sales'], name='Sales', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig_trends.update_layout(height=600, showlegend=False, title_text="Tendencias Históricas")
        fig_trends.update_xaxes(title_text="Fecha")
        fig_trends.update_yaxes(title_text="Valor")
        
        st.plotly_chart(fig_trends, use_container_width=True)
    
    with tab4:
        st.header("ℹ️ Información del Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Objetivo del Proyecto")
            st.write("""
            **Linksy** es un sistema de predicción de ventas con IA que integra:
            - Variables económicas (20 indicadores)
            - Variables del mercado (20 indicadores)  
            - Variables del producto (15 indicadores)
            
            El objetivo es predecir la demanda del mercado de un producto
            considerando todas las variables posibles sin excepciones.
            """)
            
            st.subheader("🔧 Tecnologías Utilizadas")
            st.write("""
            - **Backend:** FastAPI + Python
            - **Frontend:** Streamlit
            - **ML:** Scikit-learn, XGBoost, LightGBM
            - **Visualización:** Plotly
            - **Datos:** Pandas, NumPy
            """)
        
        with col2:
            st.subheader("📊 Variables Incluidas")
            
            st.write("**Variables Económicas (20):**")
            st.write("PIB, Inflación, Desempleo, Tasas de interés, Tipo de cambio, etc.")
            
            st.write("**Variables del Mercado (20):**")
            st.write("Precios de competidores, Cuota de mercado, Satisfacción del cliente, etc.")
            
            st.write("**Variables del Producto (15):**")
            st.write("Precio, Categoría, Características, Ciclo de vida, etc.")
            
            st.subheader("🚀 Características")
            st.write("""
            - Predicción en tiempo real
            - Múltiples algoritmos de ML
            - Análisis de importancia de características
            - Visualizaciones interactivas
            - API REST completa
            """)
        
        # Estadísticas del sistema
        if model_info:
            st.subheader("📈 Estadísticas del Modelo")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tipo de Modelo", model_info['model_type'])
            
            with col2:
                st.metric("Score del Modelo", f"{model_info['score']:.4f}")
            
            with col3:
                st.metric("Características", model_info['features_count'])

if __name__ == "__main__":
    main()

