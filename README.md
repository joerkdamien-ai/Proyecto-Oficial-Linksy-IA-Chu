# 📈 Linksy - Predictor de Ventas con IA

Un sistema completo de predicción de ventas que integra variables económicas, del mercado y del producto para predecir la demanda del mercado con alta precisión.

## 🎯 Objetivo del Proyecto

Crear un sistema de predicción de ventas que pueda predecir la demanda del mercado de un producto incluyendo **todas las variables posibles** sin excepciones:

- **20 Variables Económicas**: PIB, inflación, desempleo, tasas de interés, etc.
- **20 Variables del Mercado**: Precios de competidores, cuota de mercado, satisfacción del cliente, etc.
- **15 Variables del Producto**: Precio, categoría, características, ciclo de vida, etc.

## 🚀 Características Principales

- **Predicción en Tiempo Real**: API REST con FastAPI
- **Interfaz Web Interactiva**: Frontend con Streamlit
- **Múltiples Algoritmos ML**: Random Forest, XGBoost, LightGBM, SVM, etc.
- **Análisis de Importancia**: Identificación de variables más relevantes
- **Visualizaciones Interactivas**: Gráficos con Plotly
- **Datos Sintéticos**: Generación automática para demostración

## 🛠️ Tecnologías Utilizadas

### Backend
- **FastAPI**: API REST moderna y rápida
- **Python**: Lenguaje principal
- **Scikit-learn**: Algoritmos de machine learning
- **XGBoost & LightGBM**: Algoritmos avanzados
- **Pandas & NumPy**: Manipulación de datos

### Frontend
- **Streamlit**: Interfaz web interactiva
- **Plotly**: Visualizaciones interactivas
- **HTML/CSS**: Estilos personalizados

### Machine Learning
- **Random Forest**: Algoritmo principal
- **Gradient Boosting**: Mejora de rendimiento
- **Support Vector Regression**: Análisis no lineal
- **Validación Cruzada**: Evaluación robusta

## 📊 Variables Incluidas

### Variables Económicas (20)
- Crecimiento del PIB
- Tasa de inflación
- Tasa de desempleo
- Tasa de interés
- Tipo de cambio
- Confianza del consumidor
- Ventas al por menor
- Producción industrial
- Inicio de viviendas
- Índice del mercado de valores
- Precio del petróleo
- Precio del oro
- Rendimiento de bonos
- Oferta monetaria
- Deuda gubernamental
- Balanza comercial
- Inversión extranjera
- Crecimiento salarial
- Índice de productividad
- Confianza empresarial

### Variables del Mercado (20)
- Precio de competidores
- Cuota de mercado
- Conocimiento de marca
- Satisfacción del cliente
- Calidad del producto
- Gasto en publicidad
- Cobertura de distribución
- Factor estacional
- Actividad promocional
- Presencia online
- Engagement en redes sociales
- Reseñas de clientes
- Lanzamiento de producto
- Elasticidad del precio
- Pronóstico de demanda
- Salud de la cadena de suministro
- Entorno regulatorio
- Adopción de tecnología
- Tendencias demográficas
- Indicadores económicos

### Variables del Producto (15)
- Precio del producto
- Categoría del producto
- Características del producto
- Ciclo de vida del producto
- Nivel de innovación
- Opciones de personalización
- Período de garantía
- Servicio post-venta
- Disponibilidad del producto
- Tiempo de envío
- Política de devolución
- Agrupación de productos
- Venta cruzada
- Potencial de venta adicional
- Calificación del producto

## 🚀 Instalación y Uso

### 1. Clonar el Repositorio
```bash
git clone <repository-url>
cd linksy-sales-predictor
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar la Demostración Completa
```bash
python run_demo.py
```

Este script:
- ✅ Verifica dependencias
- 📊 Genera datos sintéticos
- 🤖 Entrena el modelo de ML
- 🚀 Inicia el servidor API
- 🌐 Abre la aplicación web

### 4. Acceder a la Aplicación

- **Aplicación Web**: http://localhost:8501
- **API REST**: http://localhost:8000
- **Documentación API**: http://localhost:8000/docs

## 📱 Uso de la Aplicación

### 1. Predicción de Ventas
1. Abre la aplicación web en http://localhost:8501
2. Ve a la pestaña "🔮 Predicción"
3. Configura las variables económicas, del mercado y del producto
4. Haz clic en "🚀 Realizar Predicción"
5. Visualiza los resultados y análisis

### 2. Análisis de Variables
- **Pestaña "📊 Análisis"**: Visualiza el impacto de las variables
- **Pestaña "📈 Tendencias"**: Analiza tendencias históricas
- **Pestaña "ℹ️ Información"**: Información del sistema

### 3. API REST
```python
import requests

# Ejemplo de predicción
payload = {
    "date": "2024-01-01",
    "economic": {
        "gdp_growth": 2.5,
        "inflation_rate": 3.0,
        # ... más variables
    },
    "market": {
        "competitor_price": 100,
        "market_share": 15,
        # ... más variables
    },
    "product": {
        "product_price": 50,
        "product_category": "Electronics",
        # ... más variables
    }
}

response = requests.post("http://localhost:8000/predict", json=payload)
prediction = response.json()
```

## 🔧 Configuración Avanzada

### Variables de Entorno
Crea un archivo `.env` para configurar:
```env
FRED_API_KEY=your_fred_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
DATABASE_URL=sqlite:///linksy.db
```

### Personalización del Modelo
Edita `config.py` para:
- Ajustar parámetros del modelo
- Agregar nuevas variables
- Modificar configuraciones

## 📈 Rendimiento del Modelo

El sistema incluye múltiples algoritmos y selecciona automáticamente el mejor:

- **Random Forest**: Robusto para datos complejos
- **XGBoost**: Excelente rendimiento
- **LightGBM**: Rápido y eficiente
- **SVM**: Bueno para relaciones no lineales

### Métricas de Evaluación
- **R² Score**: Coeficiente de determinación
- **RMSE**: Error cuadrático medio
- **MAE**: Error absoluto medio
- **Validación Cruzada**: Evaluación robusta

## 🎯 Casos de Uso

1. **Retail**: Predicción de ventas por producto
2. **E-commerce**: Optimización de inventario
3. **Manufactura**: Planificación de producción
4. **Servicios**: Predicción de demanda
5. **Análisis de Mercado**: Tendencias y oportunidades

## 🔍 Análisis de Importancia

El sistema identifica automáticamente las variables más importantes:

1. **Variables Económicas**: Impacto macroeconómico
2. **Variables del Mercado**: Competencia y demanda
3. **Variables del Producto**: Características específicas

## 📊 Visualizaciones

- **Gráficos de Predicción**: Resultados visuales
- **Análisis de Radar**: Variables económicas
- **Gráficos de Barras**: Variables del mercado
- **Tendencias Históricas**: Análisis temporal
- **Importancia de Características**: Ranking de variables

## 🚀 Despliegue

### Desarrollo Local
```bash
python run_demo.py
```

### Producción
```bash
# API
uvicorn api:app --host 0.0.0.0 --port 8000

# Streamlit
streamlit run streamlit_app.py --server.port 8501
```

## 🤝 Contribuciones

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 📞 Soporte

Para soporte técnico o preguntas:
- 📧 Email: support@linksy.com
- 💬 Issues: GitHub Issues
- 📚 Documentación: Wiki del proyecto

## 🎉 Agradecimientos

- **Scikit-learn**: Algoritmos de ML
- **Streamlit**: Framework web
- **FastAPI**: API REST
- **Plotly**: Visualizaciones
- **Pandas**: Manipulación de datos

---

**Linksy** - Predicción de Ventas con IA 🚀📈

