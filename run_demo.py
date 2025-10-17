"""
Script para ejecutar la demostración completa de Linksy
Genera datos, entrena el modelo y ejecuta la aplicación
"""
import os
import sys
import subprocess
import time
import threading
from pathlib import Path

def create_directories():
    """Crea los directorios necesarios"""
    directories = ['data', 'models', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directorio '{directory}' creado")

def generate_data():
    """Genera datos sintéticos"""
    print("🔄 Generando datos sintéticos...")
    try:
        from data_generator import DataGenerator
        generator = DataGenerator()
        data = generator.generate_all_data()
        
        # Guardar datos
        data.to_csv('data/sales_data.csv', index=False)
        print(f"✅ Datos generados: {data.shape[0]} filas, {data.shape[1]} columnas")
        return True
    except Exception as e:
        print(f"❌ Error generando datos: {str(e)}")
        return False

def train_model():
    """Entrena el modelo de ML"""
    print("🔄 Entrenando modelo de ML...")
    try:
        from ml_model import train_sales_predictor
        predictor, results = train_sales_predictor('data/sales_data.csv')
        print("✅ Modelo entrenado exitosamente")
        return True
    except Exception as e:
        print(f"❌ Error entrenando modelo: {str(e)}")
        return False

def start_api_server():
    """Inicia el servidor de la API"""
    print("🔄 Iniciando servidor API...")
    try:
        # Ejecutar API en un hilo separado
        def run_api():
            subprocess.run([sys.executable, "api.py"], check=True)
        
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        
        # Esperar a que la API esté lista
        time.sleep(5)
        print("✅ Servidor API iniciado en http://localhost:8000")
        return True
    except Exception as e:
        print(f"❌ Error iniciando API: {str(e)}")
        return False

def start_streamlit_app():
    """Inicia la aplicación Streamlit"""
    print("🔄 Iniciando aplicación Streamlit...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except Exception as e:
        print(f"❌ Error iniciando Streamlit: {str(e)}")

def check_dependencies():
    """Verifica que todas las dependencias estén instaladas"""
    print("🔄 Verificando dependencias...")
    
    required_packages = [
        'streamlit', 'fastapi', 'uvicorn', 'pandas', 'numpy',
        'scikit-learn', 'xgboost', 'lightgbm', 'matplotlib',
        'seaborn', 'plotly', 'requests', 'yfinance', 'fredapi'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Paquetes faltantes: {', '.join(missing_packages)}")
        print("💡 Ejecuta: pip install -r requirements.txt")
        return False
    
    print("✅ Todas las dependencias están instaladas")
    return True

def main():
    """Función principal"""
    print("🚀 Iniciando Linksy - Predictor de Ventas IA")
    print("=" * 50)
    
    # Verificar dependencias
    if not check_dependencies():
        return
    
    # Crear directorios
    create_directories()
    
    # Generar datos si no existen
    if not os.path.exists('data/sales_data.csv'):
        if not generate_data():
            return
    else:
        print("✅ Datos ya existen, omitiendo generación")
    
    # Entrenar modelo si no existe
    if not os.path.exists('models') or not os.listdir('models'):
        if not train_model():
            return
    else:
        print("✅ Modelo ya existe, omitiendo entrenamiento")
    
    # Iniciar API
    if not start_api_server():
        return
    
    print("\n🎉 Sistema Linksy iniciado exitosamente!")
    print("📊 API disponible en: http://localhost:8000")
    print("🌐 Aplicación web disponible en: http://localhost:8501")
    print("📚 Documentación API: http://localhost:8000/docs")
    print("\n💡 Presiona Ctrl+C para detener el sistema")
    
    try:
        # Iniciar Streamlit
        start_streamlit_app()
    except KeyboardInterrupt:
        print("\n👋 Sistema detenido por el usuario")
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")

if __name__ == "__main__":
    main()

