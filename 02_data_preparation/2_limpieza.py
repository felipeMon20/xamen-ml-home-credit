import pandas as pd
import numpy as np
import os

# --- CAMBIO DE RUTAS ---
# Ahora la entrada es el archivo que salió del paso 1
INPUT_FILE = "../data/processed/step1_merged.csv"
OUTPUT_FILE = "../data/processed/step2_clean.csv"

def limpiar_datos():
    print("\n--- 2. INICIANDO PROCESO DE LIMPIEZA ---")
    
    # Cargar datos
    if not os.path.exists(INPUT_FILE):
        print(f"Error: No encuentro {INPUT_FILE}. Ejecuta primero '1_integracion.py'")
        return
    
    df = pd.read_csv(INPUT_FILE)
    print(f"Datos integrados cargados: {df.shape}")

    # PASO 1: Eliminar columnas con demasiados nulos (> 40%)
    umbral = 0.4
    nulos = df.isnull().mean()
    columnas_a_borrar = nulos[nulos > umbral].index
    
    df_limpio = df.drop(columns=columnas_a_borrar)
    print(f"Se eliminaron {len(columnas_a_borrar)} columnas con >{umbral*100}% de nulos.")
    
    # PASO 2: Rellenar nulos restantes (Imputación)
    columnas_numericas = df_limpio.select_dtypes(include=[np.number]).columns
    columnas_categoricas = df_limpio.select_dtypes(include=['object']).columns

    print("Imputando valores faltantes (Mediana/Moda)...")
    
    # Rellenar numéricas con la Mediana
    for col in columnas_numericas:
        mediana = df_limpio[col].median()
        df_limpio[col] = df_limpio[col].fillna(mediana)
        
    # Rellenar categóricas con la Moda
    for col in columnas_categoricas:
        # Aseguramos que exista una moda antes de rellenar
        if not df_limpio[col].mode().empty:
            moda = df_limpio[col].mode()[0]
            df_limpio[col] = df_limpio[col].fillna(moda)

    # Verificar limpieza
    total_nulos = df_limpio.isnull().sum().sum()
    print(f"Total de valores nulos restantes: {total_nulos}")

    # PASO 3: Guardar el archivo limpio
    df_limpio.to_csv(OUTPUT_FILE, index=False)
    print(f"Archivo limpio guardado en: {OUTPUT_FILE}")

if __name__ == "__main__":
    limpiar_datos()