import pandas as pd
import os

# Rutas
INPUT_FILE = "../data/processed/train_clean.csv"
OUTPUT_FILE = "../data/processed/train_final.csv"

def feature_engineering():
    print("--- INICIANDO INGENIERÍA DE CARACTERÍSTICAS ---")
    
    # 1. Cargar datos limpios
    if not os.path.exists(INPUT_FILE):
        print("Falta el archivo train_clean.csv")
        return
    df = pd.read_csv(INPUT_FILE)
    
    # 2. Creación de Variables de Negocio (Requisito del examen)
    # Ratio: ¿Qué tan endeudado queda el cliente respecto a su sueldo?
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    # Ratio: ¿Qué porcentaje de su sueldo se va en la cuota anual?
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    # Duración del crédito en años (aprox)
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    
    print("Variables financieras creadas.")

    # 3. Codificación de Variables Categóricas (Encoding)
    # Convertimos texto como "M"/"F" a 0/1. 
    # Usamos get_dummies que es rápido y efectivo para este examen.
    df_final = pd.get_dummies(df, drop_first=True)
    
    # Limpiar caracteres raros en nombres de columnas (para evitar errores en algunos modelos)
    import re
    df_final = df_final.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    print(f"Texto convertido a números. Total columnas para el modelo: {df_final.shape[1]}")

    # 4. Guardar archivo final
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"¡LISTO! Dataset final guardado en: {OUTPUT_FILE}")

if __name__ == "__main__":
    feature_engineering()