# 🌍 Global Supply Chain Resilience Index (GSCRI) - MVP

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow.svg)](https://huggingface.co/)
[![Survival Analysis](https://img.shields.io/badge/Model-Cox--PH-orange.svg)](https://lifelines.readthedocs.io/)
[![Visualization](https://img.shields.io/badge/Maps-Folium-green.svg)](https://python-visualization.github.io/folium/)

## 📝 Executive Summary

**GSCRI** es un sistema inteligente de monitorización y predicción de riesgos para la cadena de suministro global. A diferencia de las herramientas tradicionales de gestión logística, GSCRI utiliza una arquitectura híbrida de **NLP (Natural Language Processing)** y **Análisis de Supervivencia (Survival Analysis)** para estimar la probabilidad de retraso en rutas comerciales marítimas en tiempo real, procesando noticias geopolíticas y datos de telemetría de buques.

El núcleo del sistema transforma titulares de prensa crudos en indicadores de riesgo cuantitativos, los cruza temporalmente con los trayectos de los buques y ajusta un modelo estadístico de **Cox Proportional Hazards** para predecir anomalías operativas.

---

## 🏗️ Technical Architecture

El flujo de datos sigue un diseño modular basado en los principios **SOLID**, permitiendo el desacoplamiento total entre la ingesta y la inferencia.

```text
[ Data Ingestion ] ----> [ NLP Intelligence ] ----> [ Feature Engineering ]
      (Mocks)            (DistilBERT SST-2)         (Time-Window Join)
                                    |                      |
                                    v                      v
[ Visualization ] <---- [ Survival Modeling ] <---- [ Master Table ]
 (Folium Map)             (CoxPH Fitter)           (Vessels + News)
```

1. **Ingestion:** Generador de datos sintéticos estocásticos que simulan telemetría real y feeds de noticias.
2. **NLP Pipeline:** Motor de inferencia basado en `DistilBert` que califica el sentimiento y aplica multiplicadores de severidad léxica ante crisis geopolíticas.
3. **Feature Engineering:** Generación del *Voyage Risk Index* cruzando dinámicamente noticias con ventanas de tiempo de tránsito específicas por buque y región.
4. **Survival Model:** Entrenamiento del algoritmo Cox-PH para identificar Hazard Ratios asociados a variables climáticas y de riesgo capturadas por el NLP.
5. **Visualization:** Renderizado geoespacial interactivo con capas de rutas, alertas críticas y un dashboard global de resiliencia.

---

## 🚀 Key Features & Innovation

### 💡 Right-Censored Data Handling

El mayor reto en logística es modelar barcos que aún no han llegado (*In Transit*). GSCRI utiliza **Análisis de Supervivencia** para tratar estos registros como datos censurados (event_observed=0). Esto permite que el modelo aprenda de los viajes en curso sin sesgar las predicciones hacia los viajes cortos ya concluidos.

### 🧠 Sentiment-Driven Risk Assessment

En lugar de listas negras de palabras estáticas, el sistema utiliza un modelo de **Transformer (HuggingFace)** para detectar la connotación real de las noticias, combinándolo con un sistema heurístico de severidad que magnifica el riesgo ante eventos como guerras, huelgas portuarias o desastres naturales.

### 📍 Geospatial Contextualization

El motor realiza un *Fuzzy Matching* regional. Si una noticia crítica ocurre en el "Canal de Suez", solo los buques cuyas rutas cruzan ese *chokepoint* recibirán un incremento asimétrico en su índice de riesgo (x2.0 penalty).

---

## 📂 Project Structure

```bash
src/
├── ingestion/       # Extracción de APIs (Mock & Real Clients)
├── processing/      # ETL, Limpieza NLP y Feature Engineering
├── models/          # Implementación de Cox-PH y Risk Index
├── visualization/   # Motor de renderizado Folium y Map Builder
data/
├── raw/             # Ingesta inicial (CSVs de simulación)
├── processed/       # Master Table y datasets enriquecidos con NLP/ML
reports/             # Outputs visuales (Global_Map.html)
```

---

## 🛠️ Setup & Usage

### Requisitos

- Python 3.10+
- PyTorch (para el pipeline de Transformers)

### Instalación

```bash
git clone https://github.com/tu-usuario/global-supply-chain-risk.git
cd global-supply-chain-risk
pip install -r requirements.txt
```

### Ejecutar Pipeline Completo

El orquestador principal genera los datos, entrena el modelo y construye el mapa visual automáticamente:

```bash
python generate_mocks.py  # Genera 1,000 barcos y 200 noticias
python run_pipeline.py    # Ejecuta NLP, Modelado y Visualización
```

---

## 🔮 Future Roadmap (Deuda Técnica Consciente)

Como todo MVP, el sistema prioriza la funcionalidad sobre la optimización extrema. El roadmap para la V2 incluye:

- **Optimización de Inferencia NLP:** Migración de loops `.iterrows()` a procesamiento por lotes (Batch Processing) en GPU para escalar a millones de noticias.
- **Enrutamiento Marítimo Real:** Integración con librerías de geometría esférica para evitar que las PolyLines crucen masas terrestres en el mapa.
- **Conectores de Producción:** Sustitución de mocks por conectores reales a **GDELT Project** (noticias globales) y **MarineTraffic API**.
- **NER Avanzado:** Implementación de Reconocimiento de Entidades Nombradas (NER) para auto-descubrimiento de locaciones no predefinidas en el diccionario.

---
**Desarrollado con enfoque en Convexidad y Resiliencia Logística.**
© 2026 GSCRI Architecture Unit.
