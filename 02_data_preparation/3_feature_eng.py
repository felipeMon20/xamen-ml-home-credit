import pandas as pd
import os
import re

# --- CAMBIO DE RUTAS ---
# Entrada: Salida del paso 2
INPUT_FILE = "../data/processed/step2_clean.csv"
# Salida: El nombre FINAL que espera el script de entrenamiento
OUTPUT_FILE = "../data/processed/train_final.csv" 

def feature_engineering():
    print("\n--- 3. INICIANDO INGENIERÍA DE CARACTERÍSTICAS ---")
    
    if not os.path.exists(INPUT_FILE):
        print("Falta el archivo step2_clean.csv")
        return
    df = pd.read_csv(INPUT_FILE)
    
    # 2. Creación de Variables de Negocio
    # --- Ratios Originales (De tu código) ---
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    
    # Si logramos cruzar con Bureau, creamos un ratio de deuda total
    if 'BUREAU_TOTAL_DEBT' in df.columns:
        # Deuda Interna (AMT_CREDIT) + Deuda Externa (BUREAU) / Ingresos
        df['TOTAL_DEBT_RATIO'] = (df['AMT_CREDIT'] + df['BUREAU_TOTAL_DEBT']) / df['AMT_INCOME_TOTAL']
        print("Nueva variable creada: TOTAL_DEBT_RATIO (Usando datos de Bureau)")
    
    print("Variables financieras calculadas.")

    # 3. Codificación (Encoding)
    df_final = pd.get_dummies(df, drop_first=True)
    
    # Limpiar nombres de columnas (Quitar caracteres especiales)
    df_final = df_final.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    print(f"Texto convertido a números. Total columnas para el modelo: {df_final.shape[1]}")

    # 4. Guardar archivo final
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"¡LISTO! Dataset final guardado en: {OUTPUT_FILE}")

if __name__ == "__main__":
    feature_engineering()