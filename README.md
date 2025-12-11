# Predicción de Riesgo de Incumplimiento de Crédito (Home Credit)

Este proyecto implementa una solución de Machine Learning end-to-end para predecir el riesgo de impago de clientes de Home Credit. El sistema sigue la metodología **CRISP-DM** y está estructurado como una arquitectura de microservicios, culminando en una API REST para la evaluación en tiempo real.

## Descripción del Problema
El objetivo es clasificar solicitudes de crédito para minimizar pérdidas financieras. Se abordaron desafíos clave como:
- **Desbalance de clases:** Solo ~8% de incumplimiento.
- **Alta dimensionalidad:** Integración y reducción de más de 120 variables.
- **Feature Engineering:** Creación de ratios financieros clave.

## Estructura del Proyecto
El código está organizado modularmente siguiendo el flujo de datos:

- `01_data_understanding/`: Análisis Exploratorio de Datos (EDA) y validación de calidad.
- `02_data_preparation/`: 
  - Limpieza de valores nulos y eliminación de variables ruidosas.
  - Ingeniería de características y codificación (One-Hot Encoding).
- `03_modeling/`: Entrenamiento del modelo (Random Forest) con manejo de desbalance (`class_weight`).
- `04_evaluation/`: Métricas de desempeño (ROC-AUC).
- `05_deployment/`: API desarrollada en **FastAPI** para servir el modelo.
- `artifacts/`: Almacenamiento de modelos serializados (`.pkl`).
- `data/`: Datos crudos y procesados (excluidos del repositorio por peso).

## Instrucciones de Ejecución

### 1. Instalación de Dependencias
```bash
pip install -r requirements.txt

2. Preparación de Datos (ETL)
El proyecto requiere procesar los datos crudos antes de entrenar. Ejecuta los scripts en orden:

  1. Limpieza de datos: Elimina columnas vacías e imputa valores nulos.
    cd 02_data_preparation
    python 1_limpieza.py

  2. Ingeniería de Características: Crea variables financieras y codifica texto a números.
    python 2_feature_eng.py
    (Regresa a la carpeta raíz ejecutando cd ..)

3. Entrenamiento del Modelo
Este script entrena el Random Forest, valida el desempeño y guarda el archivo .pkl en la carpeta artifacts.
    cd 03_modeling
    python 1_entrenamiento.py
    (Regresa a la carpeta raíz ejecutando cd ..)

4. Generación de Reportes (Opcional)
Para visualizar la Matriz de Confusión y la Curva ROC:
  Abre la carpeta 04_evaluation.
  Ejecuta el notebook reporte_modelo.ipynb en VS Code o Jupyter Notebook.

5. Despliegue de la API (Producción)
Para levantar el servidor y hacer predicciones en tiempo real:

Entra a la carpeta de despliegue:
    cd 05_deployment

Inicia el servidor con Uvicorn:
  uvicorn app:app --reload
  
Acceso a la Documentación Interactiva: Abre tu navegador y ve a: http://127.0.0.1:8000/docs

Desde allí podrás probar el endpoint /evaluate_risk enviando datos JSON.