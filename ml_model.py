"""
Modelo de Machine Learning para predicción de ventas
Incluye múltiples algoritmos y selección automática del mejor modelo
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from datetime import datetime
from config import Config

class SalesPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.best_model = None
        self.best_score = -np.inf
        self.feature_columns = []
        self.target_column = 'sales'
        
    def prepare_data(self, df):
        """Prepara los datos para el entrenamiento"""
        print("Preparando datos...")
        
        # Crear copia para no modificar el original
        data = df.copy()
        
        # Convertir fecha a características numéricas
        data['year'] = pd.to_datetime(data['date']).dt.year
        data['month'] = pd.to_datetime(data['date']).dt.month
        data['day'] = pd.to_datetime(data['date']).dt.day
        data['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek
        data['quarter'] = pd.to_datetime(data['date']).dt.quarter
        
        # Codificar variables categóricas
        categorical_columns = ['product_category', 'product_lifecycle']
        for col in categorical_columns:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.encoders[col] = le
        
        # Seleccionar características
        exclude_columns = ['date', 'sales']
        self.feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        # Separar características y objetivo
        X = data[self.feature_columns]
        y = data[self.target_column]
        
        # Manejar valores faltantes
        X = X.fillna(X.mean())
        
        return X, y
    
    def train_models(self, X, y):
        """Entrena múltiples modelos y selecciona el mejor"""
        print("Entrenando modelos...")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.MODEL_CONFIG['test_size'], 
            random_state=Config.MODEL_CONFIG['random_state']
        )
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Definir modelos
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=Config.MODEL_CONFIG['n_estimators'],
                max_depth=Config.MODEL_CONFIG['max_depth'],
                random_state=Config.MODEL_CONFIG['random_state']
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=Config.MODEL_CONFIG['n_estimators'],
                max_depth=Config.MODEL_CONFIG['max_depth'],
                learning_rate=Config.MODEL_CONFIG['learning_rate'],
                random_state=Config.MODEL_CONFIG['random_state']
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=Config.MODEL_CONFIG['n_estimators'],
                max_depth=Config.MODEL_CONFIG['max_depth'],
                learning_rate=Config.MODEL_CONFIG['learning_rate'],
                random_state=Config.MODEL_CONFIG['random_state']
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=Config.MODEL_CONFIG['n_estimators'],
                max_depth=Config.MODEL_CONFIG['max_depth'],
                learning_rate=Config.MODEL_CONFIG['learning_rate'],
                random_state=Config.MODEL_CONFIG['random_state'],
                verbose=-1
            ),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        # Entrenar y evaluar modelos
        results = {}
        
        for name, model in models.items():
            print(f"Entrenando {name}...")
            
            try:
                # Entrenar modelo
                if name in ['Linear Regression', 'Ridge', 'Lasso', 'SVR']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Evaluar modelo
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Validación cruzada
                if name in ['Linear Regression', 'Ridge', 'Lasso', 'SVR']:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': rmse,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                # Guardar importancia de características si está disponible
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(self.feature_columns, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    self.feature_importance[name] = dict(zip(self.feature_columns, abs(model.coef_)))
                
                print(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.2f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"Error entrenando {name}: {str(e)}")
                continue
        
        # Seleccionar mejor modelo
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
            self.best_model = results[best_model_name]['model']
            self.best_score = results[best_model_name]['cv_mean']
            
            print(f"\nMejor modelo: {best_model_name}")
            print(f"Score: {self.best_score:.4f}")
            
            # Guardar todos los modelos
            self.models = {name: result['model'] for name, result in results.items()}
            
        return results
    
    def optimize_hyperparameters(self, X, y):
        """Optimiza hiperparámetros del mejor modelo"""
        if self.best_model is None:
            return
        
        print("Optimizando hiperparámetros...")
        
        # Definir grillas de búsqueda
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        # Obtener nombre del mejor modelo
        best_model_name = None
        for name, model in self.models.items():
            if model == self.best_model:
                best_model_name = name
                break
        
        if best_model_name in param_grids:
            print(f"Optimizando {best_model_name}...")
            
            grid_search = GridSearchCV(
                self.best_model,
                param_grids[best_model_name],
                cv=5,
                scoring='r2',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            self.best_model = grid_search.best_estimator_
            print(f"Mejores parámetros: {grid_search.best_params_}")
            print(f"Mejor score: {grid_search.best_score_:.4f}")
    
    def predict(self, X):
        """Realiza predicciones con el mejor modelo"""
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado")
        
        # Aplicar transformaciones necesarias
        if hasattr(self.best_model, 'predict'):
            return self.best_model.predict(X)
        else:
            raise ValueError("Modelo no compatible")
    
    def get_feature_importance(self, top_n=20):
        """Obtiene la importancia de las características"""
        if not self.feature_importance:
            return {}
        
        # Obtener importancia del mejor modelo
        best_model_name = None
        for name, model in self.models.items():
            if model == self.best_model:
                best_model_name = name
                break
        
        if best_model_name and best_model_name in self.feature_importance:
            importance = self.feature_importance[best_model_name]
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_importance[:top_n])
        
        return {}
    
    def save_model(self, filepath=None):
        """Guarda el modelo entrenado"""
        if filepath is None:
            os.makedirs(Config.MODEL_PATH, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(Config.MODEL_PATH, f"sales_predictor_{timestamp}.joblib")
        
        model_data = {
            'best_model': self.best_model,
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'best_score': self.best_score
        }
        
        joblib.dump(model_data, filepath)
        print(f"Modelo guardado en: {filepath}")
        return filepath
    
    def load_model(self, filepath):
        """Carga un modelo previamente entrenado"""
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['best_model']
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.encoders = model_data['encoders']
        self.feature_columns = model_data['feature_columns']
        self.feature_importance = model_data['feature_importance']
        self.best_score = model_data['best_score']
        
        print(f"Modelo cargado desde: {filepath}")
        return True

def train_sales_predictor(data_file='data/sales_data.csv'):
    """Función principal para entrenar el predictor de ventas"""
    print("=== ENTRENANDO PREDICTOR DE VENTAS LINKSY ===")
    
    # Cargar datos
    if not os.path.exists(data_file):
        print(f"Archivo {data_file} no encontrado. Generando datos...")
        from data_generator import DataGenerator
        generator = DataGenerator()
        data = generator.generate_all_data()
        data.to_csv(data_file, index=False)
    else:
        data = pd.read_csv(data_file)
    
    print(f"Datos cargados: {data.shape}")
    
    # Inicializar predictor
    predictor = SalesPredictor()
    
    # Preparar datos
    X, y = predictor.prepare_data(data)
    print(f"Características: {X.shape[1]}, Muestras: {X.shape[0]}")
    
    # Entrenar modelos
    results = predictor.train_models(X, y)
    
    # Optimizar hiperparámetros
    predictor.optimize_hyperparameters(X, y)
    
    # Guardar modelo
    model_path = predictor.save_model()
    
    # Mostrar importancia de características
    importance = predictor.get_feature_importance()
    if importance:
        print("\n=== IMPORTANCIA DE CARACTERÍSTICAS (Top 10) ===")
        for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
            print(f"{i:2d}. {feature}: {score:.4f}")
    
    return predictor, results

if __name__ == "__main__":
    predictor, results = train_sales_predictor()

