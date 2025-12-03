import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import os

# Rutas
INPUT_FILE = "../data/processed/train_final.csv"
MODEL_PATH = "../artifacts/modelo_riesgo.pkl"

def entrenar_modelo():
    print("--- INICIANDO ENTRENAMIENTO DEL MODELO ---")
    
    # 1. Cargar datos listos
    if not os.path.exists(INPUT_FILE):
        print("Error: No encuentro el archivo train_final.csv")
        return
    
    df = pd.read_csv(INPUT_FILE)
    
    # 2. Separar características (X) y objetivo (y)
    X = df.drop(columns=['TARGET', 'SK_ID_CURR']) # Quitamos el ID y lo que queremos predecir
    y = df['TARGET']
    
    print(f"Datos cargados. Features: {X.shape[1]}")

    # 3. Split Train/Test (Validación)
    # Usamos stratify=y para mantener la misma proporción de desbalance en ambos sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Entrenar el Modelo (Random Forest)
    # IMPORTANTE: class_weight='balanced' es clave para el desbalance de este examen
    print("Entrenando Random Forest (esto puede tardar unos minutos)...")
    modelo = RandomForestClassifier(n_estimators=100, 
                                    class_weight='balanced', 
                                    random_state=42,
                                    n_jobs=-1) # Usa todos los núcleos del PC
    modelo.fit(X_train, y_train)
    
    # 5. Evaluación preliminar
    print("Entrenamiento finalizado. Evaluando...")
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f} (Un buen modelo debe ser > 0.5)")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    # 6. Guardar el modelo (Artifacts)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)
    print(f"Modelo guardado exitosamente en: {MODEL_PATH}")

if __name__ == "__main__":
    entrenar_modelo()