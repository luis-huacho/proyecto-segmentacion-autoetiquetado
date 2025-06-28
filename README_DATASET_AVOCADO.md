# 🥑 Procesamiento de Dataset de Avocados con Clasificación de Maduración

Este documento describe cómo procesar un dataset de avocados que contiene imágenes con clasificación de índice de maduración (1-5) usando el framework SDM-D modular.

## 📋 Estructura del Dataset Requerida

Tu dataset debe tener la siguiente estructura:

```
avocado_dataset/
├── description.xlsx          # Archivo Excel con información de clasificación
└── images/                   # Directorio con las imágenes
    ├── T20_d01_001_a_1.jpg
    ├── T20_d01_001_b_1.jpg
    ├── T20_d02_001_a_1.jpg
    └── ...
```

### Archivo Excel (description.xlsx)

El archivo Excel debe contener las siguientes columnas:

| Columna | Descripción | Ejemplo |
|---------|-------------|---------|
| **File Name** | Nombre del archivo de imagen (sin extensión) | T20_d01_001_a_1 |
| **Time Stamp** | Marca temporal (opcional) | 2022-04-04 18:56:55 |
| **Storage Group** | Grupo de almacenamiento (opcional) | T20 |
| **Sample** | Número de muestra (opcional) | 001 |
| **Day of Experiment** | Día del experimento (opcional) | 01 |
| **Ripening Index Classification** | **REQUERIDO**: Clasificación 1-5 | 1, 2, 3, 4, 5 |

### Clasificación de Índice de Maduración

| Índice | Etiqueta | Descripción |
|--------|----------|-------------|
| **1** | Underripe | Verde claro, firme |
| **2** | Breaking | Iniciando maduración |
| **3** | Ripe (First Stage) | Verde oscuro, listo para cosecha |
| **4** | Ripe (Second Stage) | Verde oscuro/negro, óptimo |
| **5** | Overripe | Muy oscuro, pasado del punto óptimo |

## 🚀 Uso Rápido

### Opción 1: Configuración Automática (Recomendado)

```bash
# 1. Ejecutar configuración automática
python setup_avocado_dataset.py

# 2. Seguir las instrucciones en pantalla
# El script verificará todo y creará comandos de ejemplo
```

### Opción 2: Pipeline Automático

```bash
# Procesamiento completo en un solo comando
python run_avocado_processing.py \
    --dataset_path ./avocado_dataset \
    --output_path ./results_avocado
```

### Opción 3: Paso a Paso

```bash
# 1. Preparar dataset (crear splits train/val/test)
python process_avocado_dataset.py \
    --dataset_path ./avocado_dataset \
    --output_path ./prepared_dataset

# 2. Ejecutar SDM-D con configuración optimizada
python main_sdm_modular.py \
    --image_folder ./prepared_dataset/Images/avocado \
    --output_folder ./output_avocado \
    --description_file ./prepared_dataset/description/avocado_des.txt \
    --avocado_ripening_dataset \
    --enable_visualizations \
    --box_visual \
    --color_visual \
    --save_json \
    --verbose
```

## 📁 Archivos Creados

Los nuevos archivos añadidos al proyecto son:

### 🔧 Archivos Principales

| Archivo | Propósito | Uso |
|---------|-----------|-----|
| `process_avocado_dataset.py` | Procesa Excel y crea dataset compatible | Preparación |
| `run_avocado_processing.py` | Pipeline automático completo | Procesamiento |
| `setup_avocado_dataset.py` | Configuración y verificación | Configuración |
| `main_sdm_modular.py` | **ACTUALIZADO** con soporte para avocados | Principal |

### 📝 Archivos de Configuración

| Archivo | Propósito |
|---------|-----------|
| `description/avocado_des.txt` | **ACTUALIZADO** con 5 clasificaciones |
| `README_DATASET_AVOCADO.md` | Esta documentación |

## 🎯 Funcionalidades Específicas para Avocados

### 🆕 Nuevos Argumentos en main_sdm_modular.py

```bash
# Habilitar modo dataset de avocados (optimiza parámetros automáticamente)
--avocado_ripening_dataset

# Habilitar analytics específicos para avocados
--avocado_analytics

# Configuración optimizada para avocados
--points_per_side 32        # Óptimo para avocados
--min_mask_area 100         # Filtra regiones pequeñas
--enable_nms               # Importante para avocados agrupados
--nms_threshold 0.8        # Menos estricto para formas similares
```

### 📊 Analytics Automáticos

Cuando usas `--avocado_analytics`, el sistema genera:

- **Distribución de madurez**: Gráficas de clasificación 1-5
- **Análisis de tamaño**: Correlación tamaño vs madurez  
- **Dashboard de calidad**: Score 0-100 del cultivo
- **Recomendaciones**: Basadas en distribución de madurez
- **Timeline**: Progreso temporal del procesamiento

### 🎨 Visualizaciones Específicas

- **Máscaras coloreadas** por estado de madurez
- **Cajas delimitadoras** con etiquetas de clasificación
- **Mapas de calor** de distribución espacial
- **Gráficas estadísticas** de todo el dataset

## 📊 Estructura de Salida

Después del procesamiento, tendrás:

```
results_avocado/
├── prepared_dataset/               # Dataset procesado
│   ├── Images/avocado/
│   │   ├── train/                 # 70% de las imágenes
│   │   ├── val/                   # 15% de las imágenes  
│   │   └── test/                  # 15% de las imágenes
│   ├── description/
│   │   └── avocado_des.txt        # Descripciones para SDM-D
│   └── dataset_metadata.json      # Metadatos del dataset
├── sdm_output/                     # Resultados de SDM-D
│   ├── mask/                      # Máscaras de segmentación
│   ├── labels/                    # Etiquetas YOLO
│   ├── mask_color_visual/         # Visualizaciones coloreadas
│   ├── label_box_visual/          # Cajas delimitadoras
│   ├── analytics/                 # Analytics de avocados
│   │   ├── maturity_distribution.png
│   │   ├── size_analysis.png
│   │   ├── quality_dashboard.png
│   │   └── avocado_report.json
│   └── logs/                      # Logs detallados
└── processing_summary.txt          # Resumen final
```

## 🔧 Parámetros Recomendados

### Para Avocados en Árboles

```bash
python main_sdm_modular.py \
    --points_per_side 32 \
    --min_mask_area 100 \
    --enable_nms \
    --nms_threshold 0.8 \
    --avocado_ripening_dataset
```

### Para Avocados en Primer Plano

```bash
python main_sdm_modular.py \
    --points_per_side 64 \
    --min_mask_area 200 \
    --enable_nms \
    --nms_threshold 0.9 \
    --avocado_ripening_dataset
```

## 🚨 Solución de Problemas

### Error: "Archivo Excel no encontrado"
```bash
# Verificar que el archivo se llame exactamente "description.xlsx"
ls avocado_dataset/description.xlsx

# Verificar formato del Excel (debe tener columnas específicas)
python -c "import pandas as pd; print(pd.read_excel('avocado_dataset/description.xlsx').columns.tolist())"
```

### Error: "Imágenes no encontradas"
```bash
# Verificar que las imágenes estén en el directorio correcto
ls avocado_dataset/images/*.jpg | head -5

# Verificar que los nombres coincidan con el Excel
python process_avocado_dataset.py --dataset_path avocado_dataset --output_path test_output
```

### Error: "Checkpoint SAM2 no encontrado"
```bash
# Descargar checkpoints
cd checkpoints
./download_ckpts.sh
cd ..
```

### Error: "Dependencias faltantes"
```bash
# Instalar dependencias
pip install torch torchvision opencv-python numpy pandas openpyxl matplotlib seaborn Pillow

# Para SAM2
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
```

## 📈 Optimizaciones de Rendimiento

### Para Datasets Grandes (>1000 imágenes)

```bash
# 1. Solo segmentación primero
python main_sdm_modular.py \
    --image_folder ./prepared_dataset/Images/avocado \
    --output_folder ./output_seg \
    --only_segmentation \
    --enable_nms

# 2. Anotación después  
python main_sdm_modular.py \
    --image_folder ./prepared_dataset/Images/avocado \
    --output_folder ./output_seg \
    --description_file ./prepared_dataset/description/avocado_des.txt \
    --only_annotation \
    --avocado_analytics
```

### Para Procesamiento en Lote

```bash
# Procesar múltiples subdirectorios
for subset in train val test; do
    python main_sdm_modular.py \
        --image_folder ./prepared_dataset/Images/avocado/$subset \
        --output_folder ./output_avocado/$subset \
        --description_file ./prepared_dataset/description/avocado_des.txt \
        --avocado_ripening_dataset \
        --verbose
done
```

## 🎯 Casos de Uso

### 1. Investigación Agrícola
- Análisis de maduración temporal
- Correlación entre condiciones y calidad
- Optimización de tiempo de cosecha

### 2. Control de Calidad
- Clasificación automática en línea de producción
- Detección de defectos
- Estimación de vida útil

### 3. Entrenamiento de Modelos
- Dataset balanceado para ML
- Augmentación de datos
- Validación cruzada

## 🤝 Contribuir

Para contribuir con mejoras específicas para avocados:

1. **Nuevas métricas**: Agregar a `utiles/avocado_analytics.py`
2. **Visualizaciones**: Extender `utiles/visualization_utils.py`  
3. **Optimizaciones**: Modificar parámetros en `main_sdm_modular.py`

## 📞 Soporte

Si encuentras problemas:

1. **Ejecuta primero**: `python setup_avocado_dataset.py`
2. **Revisa logs**: En `output_folder/logs/`
3. **Valida dataset**: Verifica estructura y formato Excel
4. **Prueba con subset**: Usa pocas imágenes primero

---

**✅ Compatible con Python 3.12 y optimizado para datasets de avocados con clasificación de maduración 1-5**