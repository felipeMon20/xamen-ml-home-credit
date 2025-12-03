import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# --- CONFIGURACIÓN ---
MODEL_PATH = "../artifacts/modelo_riesgo.pkl"
DATA_PATH = "../data/processed/train_final.csv"

app = FastAPI(
    title="API de Riesgo de Crédito (Home Credit)",
    description="Endpoint para predecir probabilidad de incumplimiento usando Random Forest.",
    version="1.0"
)

# 1. Cargar el modelo y las columnas de entrenamiento al iniciar
print("Cargando modelo y estructura de datos...")

if os.path.exists(MODEL_PATH):
    modelo = joblib.load(MODEL_PATH)
    print("Modelo cargado.")
else:
    print("ERROR: No se encuentra el modelo .pkl")
    modelo = None

# Cargamos solo la cabecera del CSV para saber qué columnas espera el modelo
if os.path.exists(DATA_PATH):
    df_estructura = pd.read_csv(DATA_PATH, nrows=0)
    # Eliminamos TARGET y SK_ID_CURR porque no son inputs, igual que en el entrenamiento
    columnas_modelo = [col for col in df_estructura.columns if col not in ['TARGET', 'SK_ID_CURR']]
    print(f"Estructura cargada: Se esperan {len(columnas_modelo)} columnas.")
else:
    columnas_modelo = []
    print("ERROR: No se encuentra el CSV de datos procesados.")

# 2. Definir el formato de los datos de entrada (JSON)
# Para el examen, pediremos solo los datos más importantes y rellenaremos el resto con 0
class SolicitudCredito(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    # Puedes agregar más campos aquí si quieres

@app.get("/")
def home():
    return {"mensaje": "Bienvenido a la API de Riesgo de Crédito. Ve a /docs para probarla."}

# 3. Endpoint de Predicción
@app.post("/evaluate_risk")
def predict(solicitud: SolicitudCredito):
    if not modelo:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
    
    # Paso A: Crear un DataFrame con una sola fila llena de ceros
    # Esto asegura que tengamos las 174 columnas que pide el Random Forest
    df_input = pd.DataFrame(0, index=[0], columns=columnas_modelo)
    
    # Paso B: Rellenar con los datos que envió el usuario
    datos_usuario = solicitud.dict()
    for col, val in datos_usuario.items():
        if col in df_input.columns:
            df_input[col] = val
            
    # Paso C: Generar variables calculadas (Feature Engineering en vivo)
    # Debemos repetir la lógica que usaste en el script 2_feature_eng.py
    # Si no hacemos esto, el modelo recibirá ceros en variables importantes
    df_input['CREDIT_INCOME_PERCENT'] = df_input['AMT_CREDIT'] / df_input['AMT_INCOME_TOTAL']
    df_input['ANNUITY_INCOME_PERCENT'] = df_input['AMT_ANNUITY'] / df_input['AMT_INCOME_TOTAL']
    df_input['CREDIT_TERM'] = df_input['AMT_CREDIT'] / df_input['AMT_ANNUITY']

    # Paso D: Predecir
    # predict_proba devuelve [prob_clase_0, prob_clase_1]
    probabilidad = modelo.predict_proba(df_input)[0][1]
    
    # Regla de Negocio simple para el examen
    decision = "APROBAR"
    if probabilidad > 0.50: # Umbral ajustado
        decision = "RECHAZAR"
    elif probabilidad > 0.30:
        decision = "REVISIÓN MANUAL"

    return {
        "decision": decision,
        "probabilidad_incumplimiento": round(float(probabilidad), 4),
        "mensaje": f"El cliente tiene un riesgo del {probabilidad*100:.2f}%"
    }

# Para correr: uvicorn app:app --reload