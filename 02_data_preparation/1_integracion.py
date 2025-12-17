import pandas as pd
import os

# --- CONFIGURACIÓN DE RUTAS ---
RAW_DIR = "../data/raw"
OUTPUT_FILE = "../data/processed/step1_merged.csv"

def integrar_fuentes():
    print("--- 1. INICIANDO INTEGRACIÓN DE FUENTES (Requisito: Múltiples Tablas) ---")
    
    # 1. Cargar la tabla principal (Application Train)
    train_path = os.path.join(RAW_DIR, "application_train.csv")
    if not os.path.exists(train_path):
        print(f"ERROR: No se encuentra {train_path}")
        return
    
    df_train = pd.read_csv(train_path)
    print(f"Tabla Principal cargada: {df_train.shape} filas.")

    # 2. Cargar y Agregar datos de Bureau (Historial de Crédito externo)
    # Cumplimos con el requisito de "Integración" y "Agregaciones"
    bureau_path = os.path.join(RAW_DIR, "bureau.csv")
    
    if os.path.exists(bureau_path):
        print("🔄 Procesando tabla auxiliar 'bureau.csv'...")
        df_bureau = pd.read_csv(bureau_path)
        
        # --- AGREGACIÓN ---
        # Como un cliente tiene MUCHOS créditos anteriores, debemos comprimirlos en 1 fila por cliente.
        # Calculamos: Cantidad de créditos previos, Deuda total histórica y Antigüedad promedio.
        bureau_agg = df_bureau.groupby('SK_ID_CURR').agg({
            'SK_ID_BUREAU': 'count',       # Conteo
            'AMT_CREDIT_SUM': 'sum',       # Suma Total
            'DAYS_CREDIT': 'mean'          # Promedio
        }).reset_index()
        
        # Renombrar columnas para que se entiendan en el dataset final
        bureau_agg.columns = ['SK_ID_CURR', 'BUREAU_LOAN_COUNT', 'BUREAU_TOTAL_DEBT', 'BUREAU_AVG_DAYS']
        
        # 3. UNIÓN (MERGE) - Usamos Left Join para no perder clientes de la tabla principal
        df_merged = df_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
        
        # Llenar nulos generados (si el cliente no tenía historial, ponemos 0)
        df_merged['BUREAU_LOAN_COUNT'] = df_merged['BUREAU_LOAN_COUNT'].fillna(0)
        df_merged['BUREAU_TOTAL_DEBT'] = df_merged['BUREAU_TOTAL_DEBT'].fillna(0)
        
        print(f"Integración exitosa. Columnas nuevas agregadas: {bureau_agg.shape[1]-1}")
        
    else:
        print("AVISO: No se encontró bureau.csv. Se continuará solo con train.")
        df_merged = df_train

    # 4. Guardar resultado del Paso 1
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Archivo integrado guardado en: {OUTPUT_FILE}")

if __name__ == "__main__":
    integrar_fuentes()