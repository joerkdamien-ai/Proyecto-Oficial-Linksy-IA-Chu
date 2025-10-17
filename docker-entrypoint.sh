#!/bin/bash

# Script de entrada para Docker - Linksy Sales Predictor

echo "🚀 Iniciando Linksy - Predictor de Ventas IA"
echo "=============================================="

# Crear directorios si no existen
mkdir -p data models logs

# Generar datos si no existen
if [ ! -f "data/sales_data.csv" ]; then
    echo "📊 Generando datos sintéticos..."
    python data_generator.py
fi

# Entrenar modelo si no existe
if [ ! -d "models" ] || [ -z "$(ls -A models)" ]; then
    echo "🤖 Entrenando modelo de ML..."
    python ml_model.py
fi

# Función para manejar señales
cleanup() {
    echo "👋 Deteniendo servicios..."
    kill $API_PID $STREAMLIT_PID 2>/dev/null
    exit 0
}

trap cleanup SIGTERM SIGINT

# Iniciar API en segundo plano
echo "🌐 Iniciando API en puerto 8000..."
python api.py &
API_PID=$!

# Esperar a que la API esté lista
sleep 5

# Iniciar Streamlit en segundo plano
echo "📱 Iniciando aplicación web en puerto 8501..."
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
STREAMLIT_PID=$!

# Esperar a que ambos servicios estén listos
sleep 5

echo ""
echo "✅ Sistema Linksy iniciado exitosamente!"
echo "📊 API disponible en: http://localhost:8000"
echo "🌐 Aplicación web disponible en: http://localhost:8501"
echo "📚 Documentación API: http://localhost:8000/docs"
echo ""
echo "💡 Presiona Ctrl+C para detener el sistema"

# Mantener el contenedor ejecutándose
wait

