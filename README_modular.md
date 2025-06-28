# SDM-D Framework Modular

## Descripción

Esta es una versión **desacoplada y más entendible** del framework SDM-D original. El código ha sido reorganizado en módulos independientes que son más fáciles de entender, mantener y extender.

## 🏗️ Estructura del Proyecto

```
├── segmentation.py           # Módulo principal de segmentación SAM2
├── annotations.py           # Módulo principal de anotación OpenCLIP
├── utiles/                  # Directorio de utilidades
│   ├── __init__.py
│   ├── mask_utils.py        # Procesamiento de máscaras
│   ├── clip_utils.py        # Utilidades de OpenCLIP
│   ├── label_utils.py       # Generación de etiquetas
│   ├── file_utils.py        # Manejo de archivos
│   └── visualization_utils.py # Visualizaciones
├── main_sdm_modular.py      # Script principal modular
├── ejemplo_uso_modular.py   # Ejemplos de uso independiente
└── README_MODULAR.md        # Esta documentación
```

## 🔧 Instalación

Los mismos requisitos que el SDM-D original:

```bash
# 1. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. Instalar dependencias
pip install -r requirements.txt
pip install open_clip_torch

# 3. Instalar SAM2
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..

# 4. Descargar checkpoints
cd checkpoints
chmod +x download_ckpts.sh
./download_ckpts.sh
cd ..
```

## 🚀 Uso Rápido

### Opción 1: Script Principal Modular

```bash
# Procesamiento completo (equivalente a SDM.py original)
python main_sdm_modular.py \
    --image_folder ./Images/strawberry \
    --output_folder ./output/strawberry \
    --description_file ./description/straw_des.txt \
    --enable_visualizations \
    --box_visual \
    --color_visual

# Solo segmentación
python main_sdm_modular.py \
    --image_folder ./Images/strawberry \
    --output_folder ./output/strawberry \
    --only_segmentation \
    --enable_nms

# Solo anotación (requiere máscaras existentes)
python main_sdm_modular.py \
    --image_folder ./Images/strawberry \
    --output_folder ./output/strawberry \
    --description_file ./description/straw_des.txt \
    --only_annotation \
    --color_visual
```

### Opción 2: Uso Modular Independiente

```python
# Importar módulos específicos
from segmentation import SAM2Segmentator
from annotations import CLIPAnnotator
from utiles import MaskProcessor, VisualizationManager

# Usar solo segmentación
segmentator = SAM2Segmentator()
masks = segmentator.segment_image("imagen.jpg")

# Usar solo anotación
annotator = CLIPAnnotator()
texts, labels, label_dict = annotator.load_descriptions("descriptions.txt")
predicted_label = annotator.classify_mask(image, mask, texts, labels)
```

## 📚 Módulos Principales

### 1. `segmentation.py` - Segmentación SAM2

**Clase principal:** `SAM2Segmentator`

**Funciones clave:**
- `segment_image()`: Segmenta una imagen individual
- `segment_dataset()`: Procesa un dataset completo
- Aplicación automática de Mask NMS
- Generación de visualizaciones con índices

**Ejemplo:**
```python
from segmentation import SAM2Segmentator

segmentator = SAM2Segmentator(
    checkpoint_path="./checkpoints/sam2_hiera_large.pt",
    points_per_side=32
)

masks = segmentator.segment_image("imagen.jpg")
print(f"Generadas {len(masks)} máscaras")
```

### 2. `annotations.py` - Anotación OpenCLIP

**Clase principal:** `CLIPAnnotator`

**Funciones clave:**
- `load_descriptions()`: Carga descripciones desde archivo
- `classify_mask()`: Clasifica una máscara individual  
- `annotate_dataset()`: Anota un dataset completo
- Generación automática de etiquetas YOLO

**Ejemplo:**
```python
from annotations import CLIPAnnotator

annotator = CLIPAnnotator()
texts, labels, label_dict = annotator.load_descriptions("descriptions.txt")
predicted_label = annotator.classify_mask(image, mask, texts, labels)
```

## 🛠️ Módulos de Utilidades

### `utiles/mask_utils.py` - Procesamiento de Máscaras

- **`MaskProcessor`**: Clase para manipular máscaras
  - `apply_mask_nms()`: Non-Maximum Suppression para máscaras
  - `merge_overlapping_masks()`: Fusiona máscaras superpuestas
  - `get_mask_statistics()`: Estadísticas de máscaras
  - `create_indexed_visualization()`: Visualización con índices

### `utiles/clip_utils.py` - Utilidades OpenCLIP

- **`CLIPProcessor`**: Clase para procesamiento CLIP
  - `apply_mask_to_image()`: Aplica máscara a imagen
  - `crop_object_from_background()`: Recorta objeto
  - `predict_with_clip()`: Predicción con CLIP
  - `validate_image_quality()`: Validación de calidad

### `utiles/label_utils.py` - Generación de Etiquetas

- **`LabelGenerator`**: Clase para generar etiquetas
  - `generate_yolo_polygon_label()`: Etiquetas YOLO polígono
  - `generate_yolo_bbox_label()`: Etiquetas YOLO bounding box
  - `generate_coco_annotation()`: Formato COCO
  - `validate_yolo_label()`: Validación de etiquetas

### `utiles/file_utils.py` - Manejo de Archivos

- **`FileManager`**: Clase para manejo de archivos
  - `create_output_structure()`: Crea estructura de directorios
  - `validate_dataset_integrity()`: Valida integridad del dataset
  - `get_processing_stats()`: Estadísticas de procesamiento
  - `backup_existing_output()`: Backup automático

### `utiles/visualization_utils.py` - Visualizaciones

- **`VisualizationManager`**: Clase para visualizaciones
  - `create_box_visualization()`: Visualización con cajas
  - `create_color_mask_visualization()`: Máscaras coloreadas
  - `create_comparison_visualization()`: Comparación 3 paneles
  - `create_statistics_visualization()`: Gráficos estadísticos

## 📊 Ventajas de la Versión Modular

### 🔄 **Mejor Separación de Responsabilidades**
- Cada módulo tiene una función específica y bien definida
- Fácil testing y debugging individual
- Mantenimiento simplificado

### 🧩 **Reutilización de Componentes**
- Puedes usar solo las partes que necesitas
- Fácil integración en otros proyectos
- Extensibilidad mejorada

### 📖 **Código Más Entendible**
- Funciones más pequeñas y enfocadas
- Documentación detallada en cada módulo
- Flujo de ejecución más claro

### 🔧 **Configuración Flexible**
- Parámetros específicos para cada componente
- Modo de solo segmentación o solo anotación
- Visualizaciones opcionales

### 🐛 **Debugging Facilitado**
- Errores más localizados
- Testing independiente de componentes
- Logs más específicos

## 📋 Ejemplos de Uso

### Ejemplo 1: Solo Segmentación
```python
from segmentation import SAM2Segmentator

# Configurar segmentador
segmentator = SAM2Segmentator(
    checkpoint_path="./checkpoints/sam2_hiera_large.pt",
    points_per_side=32,
    min_mask_region_area=50
)

# Procesar dataset
segmentator.segment_dataset(
    image_folder="./Images/fruits",
    output_folder="./output/masks_only",
    enable_mask_nms=True,
    save_json=True
)
```

### Ejemplo 2: Solo Clasificación
```python
from annotations import CLIPAnnotator

# Configurar clasificador
annotator = CLIPAnnotator(model_name='ViT-B-32')

# Anotar dataset existente
annotator.annotate_dataset(
    image_folder="./Images/fruits",
    mask_folder="./output/masks_only/mask",
    description_file="./descriptions/fruit_desc.txt",
    output_folder="./output/annotations",
    enable_color_visual=True
)
```

### Ejemplo 3: Pipeline Personalizado
```python
from segmentation import SAM2Segmentator
from annotations import CLIPAnnotator
from utiles import MaskProcessor, VisualizationManager

# 1. Segmentar
segmentator = SAM2Segmentator()
masks = segmentator.segment_image("imagen.jpg")

# 2. Filtrar máscaras
processor = MaskProcessor()
filtered_masks = processor.apply_mask_nms(masks)
filtered_masks = processor.filter_small_masks(filtered_masks, min_area=100)

# 3. Clasificar
annotator = CLIPAnnotator()
# ... clasificación personalizada ...

# 4. Visualizar
viz_manager = VisualizationManager()
# ... visualizaciones personalizadas ...
```

## 🔄 Equivalencias con SDM.py Original

| SDM.py Original | Versión Modular |
|----------------|-----------------|
| `SDM.py --args` | `main_sdm_modular.py --args` |
| `generate_all_sam_mask()` | `SAM2Segmentator.segment_dataset()` |
| `label_assignment()` | `CLIPAnnotator.annotate_dataset()` |
| `filter_masks_by_overlap()` | `MaskProcessor.apply_mask_nms()` |
| `clip_prediction()` | `CLIPProcessor.predict_with_clip()` |
| `box_visual()` | `VisualizationManager.create_box_visualization()` |
| `mask_color_visualization()` | `VisualizationManager.create_color_mask_visualization()` |

## ⚡ Optimizaciones y Mejoras

### 🚀 **Rendimiento**
- Procesamiento más eficiente de memoria
- Limpieza automática de caché GPU
- Procesamiento en lotes optimizado

### 🛡️ **Robustez**
- Manejo de errores más granular
- Validación de entrada mejorada
- Recuperación automática de fallos

### 📊 **Monitoreo**
- Progreso de procesamiento detallado
- Estadísticas en tiempo real
- Reportes automáticos

### 🔧 **Configuración**
- Validación de argumentos
- Configuración por módulo
- Modos de operación flexibles

## 🧪 Testing y Debugging

### Ejecutar Ejemplos
```bash
# Ejecutar todos los ejemplos
python ejemplo_uso_modular.py

# Testing individual de módulos (en Python)
from utiles import MaskProcessor
processor = MaskProcessor()
# ... testing específico ...
```

### Debug de Componentes Individuales
```python
# Debug de segmentación
from segmentation import SAM2Segmentator
segmentator = SAM2Segmentator()
masks = segmentator.segment_image("test_image.jpg")

# Debug de clasificación
from annotations import CLIPAnnotator
annotator = CLIPAnnotator()
debug_info = annotator.clip_processor.debug_classification(
    model, image_input, texts, labels, "test_image.jpg"
)
print(debug_info)
```

## 📝 Archivos de Configuración

### Estructura de Descripciones (`description/straw_des.txt`)
```
a red strawberry with numerous points, ripe
a pale green strawberry with numerous points, unripe
a green veined strawberry leaf, leaf
a long and thin stem, stem
a white flower, flower
soil or background or something else, others
```

### Argumentos del Script Principal
```bash
# Argumentos mínimos requeridos
--image_folder ./Images/strawberry
--output_folder ./output/strawberry

# Para anotación (requerido si no es --only_segmentation)
--description_file ./description/straw_des.txt

# Configuración de modelos
--sam2_checkpoint ./checkpoints/sam2_hiera_large.pt
--clip_model ViT-B-32
--clip_pretrained laion2b_s34b_b79k

# Opciones de procesamiento
--enable_nms                 # Aplicar NMS a máscaras
--nms_threshold 0.9          # Umbral NMS
--points_per_side 32         # Puntos SAM2
--min_mask_area 50           # Área mínima máscara

# Opciones de salida
--save_json                  # Guardar metadatos JSON
--enable_visualizations      # Generar visualizaciones
--box_visual                 # Visualización con cajas
--color_visual               # Visualización coloreada

# Modos de operación
--only_segmentation          # Solo segmentar
--only_annotation            # Solo anotar
--backup_existing            # Crear backup
--verbose                    # Información detallada
```

## 🔄 Migración desde SDM.py Original

### Para usuarios existentes:

1. **Reemplazar llamada simple:**
```bash
# Antes
python SDM.py --image_folder ./Images/strawberry --out_folder ./output/strawberry --des_file ./description/straw_des.txt

# Ahora
python main_sdm_modular.py --image_folder ./Images/strawberry --output_folder ./output/strawberry --description_file ./description/straw_des.txt
```

2. **Aprovechar nuevas funcionalidades:**
```bash
# Nuevas opciones disponibles
python main_sdm_modular.py \
    --image_folder ./Images/strawberry \
    --output_folder ./output/strawberry \
    --description_file ./description/straw_des.txt \
    --backup_existing \
    --verbose \
    --enable_visualizations \
    --box_visual \
    --color_visual
```

3. **Uso programático:**
```python
# Antes: tenías que usar SDM.py como script
# Ahora: puedes usar módulos independientes
from segmentation import SAM2Segmentator
from annotations import CLIPAnnotator

# Tu código personalizado aquí...
```

## 🤝 Contribución

### Estructura para Nuevas Funcionalidades

1. **Nuevo procesador de máscaras:**
   - Agregar método a `utiles/mask_utils.py`
   - Seguir patrón de la clase `MaskProcessor`

2. **Nueva visualización:**
   - Agregar método a `utiles/visualization_utils.py`
   - Seguir patrón de la clase `VisualizationManager`

3. **Nuevo formato de etiquetas:**
   - Agregar método a `utiles/label_utils.py`
   - Seguir patrón de la clase `LabelGenerator`

### Ejemplo de Extensión
```python
# En utiles/mask_utils.py
class MaskProcessor:
    def new_filtering_method(self, masks, custom_param):
        """Nueva funcionalidad de filtrado"""
        # Tu implementación aquí
        return filtered_masks

# En tu código
from utiles import MaskProcessor
processor = MaskProcessor()
filtered_masks = processor.new_filtering_method(masks, param_value)
```

## 📞 Soporte

- **Issues:** Reportar problemas específicos del módulo
- **Documentación:** Cada módulo tiene documentación inline
- **Ejemplos:** Ver `ejemplo_uso_modular.py` para casos de uso

## 🏁 Conclusión

Esta versión modular del framework SDM-D ofrece:

✅ **Mayor flexibilidad** - Usa solo lo que necesitas  
✅ **Mejor mantenibilidad** - Código organizado y documentado  
✅ **Fácil extensión** - Agrega nuevas funcionalidades fácilmente  
✅ **Testing independiente** - Debuggea componentes por separado  
✅ **Reutilización** - Integra módulos en otros proyectos  

¡Disfruta explorando las nuevas posibilidades del framework SDM-D modular! 🚀