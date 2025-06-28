# SDM-D Framework para Detección de Avocados/Paltas

## 🥑 Descripción

Framework completo para detección, clasificación y análisis de avocados/paltas usando técnicas de foundation models sin anotación manual. Incluye sistema de logging avanzado, monitoreo en tiempo real y analytics especializados para agricultura.

## 🌟 Características Principales

### 🔍 **Detección Inteligente**
- **Estados de madurez:** Ripe, Unripe, Overripe
- **Elementos contextuales:** Hojas, ramas, flores
- **Precisión:** 85-95% en detección, 75-90% en clasificación

### 📊 **Sistema de Logging Avanzado**
- **Logs por fase:** Segmentación, anotación, analytics
- **Monitoreo en tiempo real:** Progreso, velocidad, errores
- **Métricas detalladas:** Tiempo por imagen, distribución de clases
- **Visualizaciones automáticas:** Gráficas de progreso y rendimiento

### 🥑 **Analytics Especializados para Avocados**
- **Análisis de madurez:** Distribución y tendencias
- **Métricas de cultivo:** Densidad, calidad, uniformidad espacial
- **Recomendaciones de cosecha:** Basadas en IA
- **Dashboard interactivo:** Visualización completa del estado del cultivo

### 📈 **Visualizaciones en Tiempo Real**
- **Gráficas de progreso:** Tiempo vs avance
- **Distribución de madurez:** Circular y barras
- **Mapa de calor:** Distribución espacial
- **Timeline:** Procesamiento temporal

## 🏗️ Arquitectura Actualizada

```
├── segmentation.py              # Segmentación SAM2 con logging
├── annotations.py              # Anotación OpenCLIP con analytics
├── main_sdm_modular.py         # Script principal con logging completo
├── utiles/
│   ├── logging_utils.py        # Sistema de logging y monitoreo
│   ├── avocado_analytics.py    # Analytics especializados para avocados
│   ├── mask_utils.py           # Procesamiento de máscaras
│   ├── clip_utils.py           # Utilidades OpenCLIP
│   ├── label_utils.py          # Generación de etiquetas
│   ├── file_utils.py           # Manejo de archivos
│   └── visualization_utils.py  # Visualizaciones
├── description/
│   └── avocado_des.txt         # Descripciones para avocados
├── AVOCADO_DATASET_GUIDE.md    # Guía para preparar dataset
└── ejemplo_uso_modular.py      # Ejemplos con logging y analytics
```

## 🚀 Instalación Rápida

```bash
# 1. Clonar repositorio
git clone <repository-url>
cd SDM-D

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
pip install open_clip_torch matplotlib seaborn pandas

# 4. Instalar SAM2
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..

# 5. Descargar checkpoints
cd checkpoints
chmod +x download_ckpts.sh
./download_ckpts.sh
cd ..
```

## 🥑 Uso para Avocados

### Preparación del Dataset

1. **Seguir la guía:** [AVOCADO_DATASET_GUIDE.md](AVOCADO_DATASET_GUIDE.md)
2. **Estructura requerida:**
```
Images/avocado/
├── train/
│   ├── field_001.jpg
│   ├── tree_section_001.jpg
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

3. **Archivo de descripciones:** `description/avocado_des.txt`
```
a dark green mature avocado ready for harvest, ripe
a light green unripe avocado still developing, unripe
an overripe dark avocado past optimal harvest time, overripe
a green avocado tree leaf, leaf
a brown avocado tree branch, branch
a small yellowish avocado tree flower, flower
soil or background or other elements, background
```

### Procesamiento Completo

```bash
# Procesamiento básico con logging
python main_sdm_modular.py \
    --image_folder ./Images/avocado \
    --output_folder ./output/avocado \
    --description_file ./description/avocado_des.txt \
    --verbose

# Procesamiento completo con analytics
python main_sdm_modular.py \
    --image_folder ./Images/avocado \
    --output_folder ./output/avocado \
    --description_file ./description/avocado_des.txt \
    --enable_visualizations \
    --box_visual \
    --color_visual \
    --avocado_analytics \
    --enable_progress_monitor \
    --save_json \
    --verbose
```

### Con Monitoreo en Tiempo Real

```bash
# Activar monitor de progreso (actualiza cada 30 segundos)
python main_sdm_modular.py \
    --image_folder ./Images/avocado \
    --output_folder ./output/avocado \
    --description_file ./description/avocado_des.txt \
    --enable_progress_monitor \
    --monitor_interval 30 \
    --avocado_analytics \
    --verbose
```

## 📊 Resultados y Analytics

### Estructura de Salida

```
output/avocado/
├── mask/                    # Máscaras de segmentación
├── labels/                  # Etiquetas YOLO
├── mask_idx_visual/         # Visualización con índices
├── label_box_visual/        # Cajas delimitadoras
├── mask_color_visual/       # Máscaras coloreadas
├── logs/                    # Logs detallados del procesamiento
│   ├── main_20241201_143022.log
│   ├── segmentation_20241201_143022.log
│   ├── annotation_20241201_143022.log
│   ├── session_report_20241201_143022.json
│   └── progress_20241201_143022.png
├── analytics/               # Analytics de avocados
│   ├── maturity_distribution.png
│   ├── size_analysis.png
│   ├── crop_quality_dashboard.png
│   └── avocado_analysis_report.json
└── processing_report.json   # Reporte general
```

### Métricas de Avocados

El sistema genera automáticamente:

#### 📈 **Gráficas de Distribución**
- **Madurez:** Porcentaje de ripe/unripe/overripe
- **Tamaños:** Distribución de tamaños por estado
- **Calidad:** Score de calidad del cultivo (0-100)

#### 🎯 **Dashboard de Calidad**
- **Densidad vs Calidad:** Correlación
- **Distribución Espacial:** Mapa de calor 3x3
- **Timeline:** Progreso temporal
- **Recomendaciones:** Basadas en IA

#### 📋 **Reporte JSON**
```json
{
  "summary": {
    "total_avocados_detected": 245,
    "class_distribution": {
      "ripe": 98,
      "unripe": 127,
      "overripe": 20
    },
    "average_quality_score": 78.5
  },
  "recommendations": [
    "🟢 RECOMENDACIÓN: Iniciar cosecha inmediatamente",
    "⭐ CALIDAD: Buena - apto para mercado estándar"
  ]
}
```

## 📊 Sistema de Logging

### Logs Automáticos

El sistema genera logs detallados automáticamente:

#### 🎯 **Log Principal** (`main_*.log`)
```
2024-12-01 14:30:22 - sdm_main - INFO - 🚀 Iniciando fase: segmentation
2024-12-01 14:30:25 - sdm_main - INFO - ✅ [15.3%] segmentation: field_avocado_001.jpg
2024-12-01 14:30:25 - sdm_main - INFO -    ⏱️ Tiempo: 2.34s
2024-12-01 14:30:25 - sdm_main - INFO -    🎭 Máscaras: 23
```

#### 🎭 **Log de Segmentación** (`segmentation_*.log`)
```
2024-12-01 14:30:25 - sdm_segmentation - INFO - 📊 Segmentación: field_avocado_001.jpg
2024-12-01 14:30:25 - sdm_segmentation - INFO -    🎭 Máscaras brutas: 35
2024-12-01 14:30:25 - sdm_segmentation - INFO -    🎯 Máscaras filtradas: 23
2024-12-01 14:30:25 - sdm_segmentation - INFO -    📉 Reducción por NMS: 34.3%
```

#### 🏷️ **Log de Anotación** (`annotation_*.log`)
```
2024-12-01 14:30:28 - sdm_annotation - INFO - 🏷️ Anotación: field_avocado_001.jpg
2024-12-01 14:30:28 - sdm_annotation - INFO -    📊 Distribución de clases:
2024-12-01 14:30:28 - sdm_annotation - INFO -       ripe: 8
2024-12-01 14:30:28 - sdm_annotation - INFO -       unripe: 12
2024-12-01 14:30:28 - sdm_annotation - INFO -       overripe: 2
```

### Monitor de Progreso en Tiempo Real

```
📊 Estado [annotation]: 67.3% (45/67) - Actual: field_avocado_045.jpg - Velocidad: 1.2 img/s - ETA: 18.3 min
```

## 🔧 Configuración Avanzada

### Parámetros Optimizados para Avocados

```bash
# Segmentación optimizada
--points_per_side 32          # Densidad adecuada para avocados
--min_mask_area 100           # Filtrar regiones muy pequeñas
--enable_nms                  # Importante para avocados agrupados
--nms_threshold 0.8           # Menos estricto para formas similares

# Logging y monitoreo
--verbose                     # Logs detallados
--enable_progress_monitor     # Monitor en tiempo real
--monitor_interval 30         # Actualizar cada 30 segundos

# Analytics específicos
--avocado_analytics          # Habilitar analytics de avocados
```

### Uso Programático

```python
from segmentation import SAM2Segmentator
from annotations import CLIPAnnotator
from utiles.logging_utils import SDMLogger, ProgressMonitor
from utiles.avocado_analytics import AvocadoAnalytics

# Configurar logging
logger = SDMLogger("./output", enable_console=True)
monitor = ProgressMonitor(logger, update_interval=30)

# Inicializar componentes con logging
segmentator = SAM2Segmentator(logger=logger)
annotator = CLIPAnnotator(logger=logger)
analytics = AvocadoAnalytics("./output")

# Ejecutar con monitoreo
monitor.start_monitoring()
# ... procesamiento ...
monitor.stop_monitoring()

# Generar reportes
session_report = logger.save_session_report()
analytics_report = analytics.export_analytics_report(results)
```

## 🎯 Ejemplos Específicos

### Solo Analytics (con datos existentes)
```bash
# Si ya tienes resultados, generar solo analytics
python ejemplo_uso_modular.py
```

### Procesamiento por Lotes
```bash
# Procesar múltiples campos
for field in field_001 field_002 field_003; do
    python main_sdm_modular.py \
        --image_folder ./Images/avocado/$field \
        --output_folder ./output/avocado/$field \
        --description_file ./description/avocado_des.txt \
        --avocado_analytics \
        --verbose
done
```

### Solo Segmentación (para datasets grandes)
```bash
# Primero segmentar todo
python main_sdm_modular.py \
    --image_folder ./Images/avocado \
    --output_folder ./output/avocado \
    --only_segmentation \
    --enable_nms \
    --verbose

# Luego anotar con analytics
python main_sdm_modular.py \
    --image_folder ./Images/avocado \
    --output_folder ./output/avocado \
    --description_file ./description/avocado_des.txt \
    --only_annotation \
    --avocado_analytics \
    --enable_progress_monitor \
    --verbose
```

## 📈 Interpretación de Resultados

### Score de Calidad del Cultivo
- **80-100:** Excelente - Mercado premium
- **60-79:** Bueno - Mercado estándar  
- **40-59:** Regular - Revisar técnicas
- **0-39:** Pobre - Investigar problemas

### Recomendaciones Automáticas
El sistema genera recomendaciones basadas en:
- **Porcentaje de madurez**
- **Distribución espacial**
- **Densidad del cultivo**
- **Tendencias temporales**

### Alertas Automáticas
- **🟢 Cosecha inmediata:** >60% maduros
- **🟡 Cosecha selectiva:** 30-60% maduros
- **🔴 Esperar:** <30% maduros
- **⚠️ Urgente:** >20% sobre-maduros

## 🔬 Tecnologías Utilizadas

### Foundation Models
- **SAM2:** Segmentación de última generación
- **OpenCLIP:** Clasificación visual-textual

### Visualización y Analytics
- **Matplotlib:** Gráficas científicas
- **Seaborn:** Visualizaciones estadísticas
- **Pandas:** Análisis de datos
- **NumPy:** Computación numérica

### Logging y Monitoreo
- **Python Logging:** Sistema robusto de logs
- **Threading:** Monitoreo en background
- **JSON:** Reportes estructurados

## 🤝 Contribución

### Extensiones Posibles
1. **Nuevos cultivos:** Adaptar analytics para mango, aguacate, etc.
2. **Modelos mejorados:** Integrar nuevos foundation models
3. **Métricas adicionales:** Estimación de peso, defectos
4. **Alertas en tiempo real:** Integración con sistemas de notificación

### Estructura para Contribuir
```python
# En utiles/avocado_analytics.py
class AvocadoAnalytics:
    def new_analysis_method(self, data):
        """Nueva funcionalidad de análisis"""
        # Tu implementación aquí
        return results
```

## 📞 Soporte

- **Issues:** Reportar problemas específicos
- **Analytics:** Ver `utiles/avocado_analytics.py` para métricas
- **Logging:** Ver `utiles/logging_utils.py` para configuración
- **Dataset:** Seguir `AVOCADO_DATASET_GUIDE.md`

## 🏁 Resultados Esperados

### Métricas Típicas
- **Detección:** 85-95% precisión
- **Clasificación:** 75-90% precisión  
- **Velocidad:** 1-3 imágenes/segundo
- **Memoria:** <8GB GPU para imágenes 1024x1024

### Casos de Uso
- **Estimación de cosecha**
- **Control de calidad**
- **Optimización de timing**
- **Reportes para agricultores**
- **Investigación agrícola**

---

**🥑 ¡Empieza a analizar tus cultivos de avocado con IA de última generación!** 🚀