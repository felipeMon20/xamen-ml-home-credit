# Predicción de Riesgo de Incumplimiento de Crédito (Home Credit)

Este proyecto implementa una solución de **Machine Learning end-to-end** para predecir el riesgo de impago de clientes de **Home Credit**. El sistema sigue la metodología **CRISP-DM** y está estructurado bajo una **arquitectura de microservicios**, culminando en una **API REST** para la evaluación de solicitudes de crédito en tiempo real.

---

## Descripción del Problema

El objetivo principal es **clasificar solicitudes de crédito** para minimizar las pérdidas financieras asociadas al incumplimiento de pago.

Durante el desarrollo se abordaron los siguientes desafíos técnicos clave:

* **Desbalance de clases**: La clase minoritaria (incumplimiento) representa aproximadamente el 8% de los registros.
* **Integración de fuentes de datos**: Enriquecimiento del dataset principal mediante el cruce con el historial de créditos externos (`bureau.csv`).
* **Feature Engineering**: Creación de ratios financieros y variables agregadas relevantes para el modelo.

---

## Estructura del Proyecto

El proyecto está organizado de forma modular, siguiendo el flujo de datos definido por CRISP-DM:

```
├── 01_data_understanding/
│   └── Notebooks de Análisis Exploratorio de Datos (EDA)
│
├── 02_data_preparation/
│   ├── 1_integracion.py      # Integración application_train + bureau
│   ├── 2_limpieza.py         # Limpieza de nulos y reducción de dimensionalidad
│   └── 3_feature_eng.py      # Creación de variables financieras y encoding
│
├── 03_modeling/
│   └── 1_entrenamiento.py    # Entrenamiento del modelo Random Forest
│
├── 04_evaluation/
│   └── reporte_modelo.ipynb  # Evaluación: Matriz de Confusión y ROC-AUC
│
├── 05_deployment/
│   └── app.py                # API REST desarrollada con FastAPI
│
├── artifacts/
│   └── modelo_rf.pkl         # Modelo entrenado y serializado
│
├── data/
│   └── (datos crudos y procesados, excluidos del repositorio)
│
├── requirements.txt
└── README.md
```

---

## Instrucciones de Ejecución

Sigue los siguientes pasos **en orden estricto** para reproducir el flujo completo del proyecto.

---

### 1. Instalación de Dependencias

Instala las librerías necesarias ejecutando:

```bash
pip install -r requirements.txt
```

---

### 2. Preparación de Datos (ETL)

El procesamiento de datos se divide en **tres etapas obligatorias**.

#### 2.1 Integración de Fuentes

Une la tabla principal con el historial de crédito externo:

```bash
cd 02_data_preparation
python 1_integracion.py
```

#### 2.2 Limpieza de Datos

Elimina columnas con alto porcentaje de nulos e imputa valores faltantes:

```bash
python 2_limpieza.py
```

#### 2.3 Ingeniería de Características

Crea los ratios financieros y variables finales utilizadas por el modelo:

```bash
python 3_feature_eng.py
```

Regresa a la carpeta raíz:

```bash
cd ..
```

---

### 3. Entrenamiento del Modelo

Entrena el modelo **Random Forest**, valida su desempeño y guarda el modelo serializado en la carpeta `artifacts`:

```bash
cd 03_modeling
python 1_entrenamiento.py
cd ..
```

---

### 4. Generación de Reportes (Opcional)

Para visualizar el desempeño del modelo:

1. Ingresa a la carpeta `04_evaluation`.
2. Abre el archivo `reporte_modelo.ipynb` en **VS Code** o **Jupyter Notebook**.

El notebook incluye:

* Matriz de Confusión
* Curva ROC
* Métrica ROC-AUC

---

### 5. Despliegue de la API (Producción)

Para levantar el servicio de predicción en tiempo real:

1. Ingresa a la carpeta de despliegue:

```bash
cd 05_deployment
```

2. Inicia el servidor con **Uvicorn**:

```bash
uvicorn app:app --reload
```

La API quedará disponible para recibir solicitudes de predicción de riesgo crediticio.
