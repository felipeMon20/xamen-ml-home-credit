import pandas as pd
import numpy as np
import os

# 1. Configuración de rutas
INPUT_FILE = "../data/raw/application_train.csv"
OUTPUT_FILE = "../data/processed/train_clean.csv"

def limpiar_datos():
    print("--- INICIANDO PROCESO DE LIMPIEZA ---")
    
    # Cargar datos
    if not os.path.exists(INPUT_FILE):
        print(f"Error: No encuentro {INPUT_FILE}")
        return
    
    df = pd.read_csv(INPUT_FILE)
    print(f"Datos originales cargados: {df.shape}")

    # PASO 1: Eliminar columnas con demasiados nulos (> 40%)
    # Esto ayuda a reducir la dimensionalidad (requisito del examen)
    umbral = 0.4
    nulos = df.isnull().mean()
    columnas_a_borrar = nulos[nulos > umbral].index
    
    df_limpio = df.drop(columns=columnas_a_borrar)
    print(f"Se eliminaron {len(columnas_a_borrar)} columnas con >{umbral*100}% de nulos.")
    
    # PASO 2: Rellenar nulos restantes (Imputación)
    # Variables numéricas -> Rellenar con la Mediana
    # Variables de texto (categóricas) -> Rellenar con la Moda (el valor más común)
    
    columnas_numericas = df_limpio.select_dtypes(include=[np.number]).columns
    columnas_categoricas = df_limpio.select_dtypes(include=['object']).columns

    print("Imputando valores faltantes...")
    
    # Rellenar numéricas
    for col in columnas_numericas:
        mediana = df_limpio[col].median()
        df_limpio[col] = df_limpio[col].fillna(mediana) # Versión segura para Pandas nuevos
        
    # Rellenar categóricas
    for col in columnas_categoricas:
        moda = df_limpio[col].mode()[0]
        df_limpio[col] = df_limpio[col].fillna(moda)

    # Verificar que no queden nulos
    total_nulos = df_limpio.isnull().sum().sum()
    print(f"Total de valores nulos restantes: {total_nulos}")
    print(f"Dimensiones finales: {df_limpio.shape}")

    # PASO 3: Guardar el archivo limpio
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_limpio.to_csv(OUTPUT_FILE, index=False)
    print(f"Archivo guardado exitosamente en: {OUTPUT_FILE}")

if __name__ == "__main__":
    limpiar_datos()