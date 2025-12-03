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