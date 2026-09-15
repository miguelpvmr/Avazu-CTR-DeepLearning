# Predicción de Clics en Publicidad Digital

## Contexto

El Click-Through Rate (CTR) mide la proporción de impresiones publicitarias que resultan en un clic por parte del usuario. Los sistemas de subasta en tiempo real y búsqueda patrocinada utilizan esta métrica como insumo para asignar presupuesto, priorizar inventario y personalizar la entrega de anuncios. Su dinámica depende del comportamiento del usuario, del contexto de navegación, del dispositivo y del inventario disponible, lo que produce patrones de interacción heterogéneos y altamente dependientes de variables categóricas de alta cardinalidad.

Este proyecto compara el desempeño de redes neuronales multicapa (*Multilayer Perceptron*, MLP) implementadas en dos entornos —scikit-learn y PySpark— para predecir la ocurrencia de un clic sobre anuncios móviles. La interpretabilidad local se aborda mediante LIME (*Local Interpretable Model-agnostic Explanations*) para analizar predicciones individuales del modelo.

## Objetivos

- **Modelado con scikit-learn:** Entrenar un `MLPClassifier` sobre una muestra representativa del conjunto de datos, con búsqueda de hiperparámetros mediante `GridSearchCV`.
- **Modelado con PySpark:** Entrenar un `MultilayerPerceptronClassifier` sobre el conjunto completo, con búsqueda de hiperparámetros sobre la arquitectura de capas y la tasa de aprendizaje.
- **Evaluación comparativa:** Contrastar ambos entornos mediante métricas de clasificación (F1-score, AUC-ROC, recall, precisión) y tiempos de entrenamiento y predicción.
- **Interpretabilidad local:** Aplicar LIME sobre instancias mal clasificadas para identificar las variables con mayor influencia en la decisión del modelo.

## Fuentes de datos

El conjunto de datos proviene de la competencia *Avazu Click-Through Rate Prediction* alojada en Kaggle. Comprende aproximadamente 40 millones de impresiones publicitarias recopiladas durante diez días consecutivos de octubre de 2014, cada una con información anonimizada asociada con el anuncio, el sitio o aplicación, el dispositivo del usuario y el contexto temporal de la impresión.

El período abarcado permite capturar variaciones en el comportamiento de los usuarios a lo largo del ciclo diario. La cobertura temporal no alcanza un ciclo mensual o anual completo, por lo que las componentes de mes y año no aportan variabilidad al modelo. Las variables incluyen identificadores de alta cardinalidad (`site_id`, `app_id`, `device_id`, `device_ip`), variables categóricas anonimizadas (`C1`, `C14`–`C21`), indicadores de posición y tipo de dispositivo (`banner_pos`, `device_type`, `device_conn_type`) y la marca temporal en formato `YYMMDDHH`.

El conjunto de prueba (`test.gz`) incluido en el repositorio original no fue considerado en este proyecto.

**Referencia:** S. Wang y W. Cukierski, "Click-Through Rate Prediction," Kaggle, 2014. [En línea]. Disponible: https://www.kaggle.com/competitions/avazu-ctr-prediction/overviewcompetitions/avazu-ctr-prediction

## Instalación y configuración

Para configurar el entorno de desarrollo y comenzar a trabajar, puede seguir los siguientes pasos:

**1. Clonar el repositorio:**

```bash
git clone https://github.com/miguelpvmr/Avazu-CTR-DeepLearning.git
cd Avazu-CTR-DeepLearning
```

**2. Recrear el entorno de Conda:**

```bash
conda env create -f environment.yml
```

**3. Activar el entorno:**

```bash
conda activate avazu-ctr
```

## Estructura del proyecto

El repositorio utiliza una arquitectura modular para la gestión de datos, el desarrollo de notebooks analíticos y el código fuente (`src`), junto con la configuración de MyST Markdown y la automatización mediante GitHub Actions:

```bash
.
├── .github/
│   └── workflows/
│       └── deploy.yaml
├── data/
│   ├── predictions/
│   │   ├── mlp_100_50_pipeline_predictions.parquet
│   │   └── mlp_100_pipeline_predictions.parquet
│   ├── raw/
│   │   ├── train.gz
│   │   └── train.parquet
│   └── processed/
│       ├── train.parquet
│       └── test.parquet
├── models/
│   ├── sklearn/
│   └── pyspark/
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_sklearn_mlp_training.ipynb
│   ├── 03_spark_mlp_training.ipynb
│   └── 04_mlp_evaluation.ipynb
├── src/
│   ├── trainers/
│   │   ├── pyspark_trainer.py
│   │   └── sklearn_trainer.py
│   └── utils/
│       ├── build_features.py
│       ├── convert_to_parquet.py
│       ├── duckdb_eda_toolkit.py
│       └── evaluation_toolkit.py
├── custom.css
├── environment.yml
├── LICENSE
├── myst.yaml
├── README.md
└── setup.cfg
```

## Licencia

Distribuido bajo la Licencia MIT. Esta licencia permite el uso, modificación y distribución del código con pocas restricciones, siempre que se incluya el aviso de copyright y la declaración de permisos. Para más detalles, consulte el archivo `LICENSE`.
