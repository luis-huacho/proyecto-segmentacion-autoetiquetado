# Guía para Preparar Dataset de Avocados/Paltas

## 📋 Descripción General

Esta guía detalla cómo preparar un dataset de imágenes de avocados/paltas para usar con el framework SDM-D. El sistema está optimizado para detectar y clasificar avocados en diferentes estados de madurez.

## 🎯 Estados de Madurez Detectables

El sistema puede identificar los siguientes estados de avocados:

### 🟢 **Avocados Maduros (ripe)**
- **Características:** Color verde oscuro a negro, superficie ligeramente blanda
- **Descripción para IA:** "a dark green mature avocado ready for harvest"
- **Uso:** Listos para cosecha inmediata
- **Valor comercial:** Alto

### 🟡 **Avocados Inmaduros (unripe)**  
- **Características:** Color verde claro a medio, superficie firme
- **Descripción para IA:** "a light green unripe avocado still developing"
- **Uso:** Necesitan más tiempo de maduración
- **Valor comercial:** Medio (para maduración controlada)

### 🟤 **Avocados Sobre-maduros (overripe)**
- **Características:** Color muy oscuro/negro, superficie muy blanda, posibles manchas
- **Descripción para IA:** "an overripe dark avocado past optimal harvest time"
- **Uso:** Cosecha urgente, procesamiento industrial
- **Valor comercial:** Bajo

### 🌿 **Elementos Adicionales Detectables**
- **Hojas (leaf):** "a green avocado tree leaf"
- **Ramas/Tallos (branch):** "a brown avocado tree branch"
- **Flores (flower):** "a small yellowish avocado tree flower"
- **Fondo (background):** "soil or background or other elements"

## 📁 Estructura del Dataset

### Organización Recomendada

```
Images/avocado/
├── train/
│   ├── field_001.jpg
│   ├── field_002.jpg
│   ├── tree_section_001.jpg
│   └── ...
├── val/
│   ├── validation_001.jpg
│   ├── validation_002.jpg
│   └── ...
└── test/
    ├── test_001.jpg
    ├── test_002.jpg
    └── ...
```

### Distribución Sugerida
- **Train:** 70% de las imágenes
- **Validation:** 15% de las imágenes  
- **Test:** 15% de las imágenes

## 📸 Especificaciones de Imágenes

### Requisitos Técnicos
- **Formato:** JPG, PNG
- **Resolución mínima:** 640x640 píxeles
- **Resolución recomendada:** 1024x1024 o superior
- **Calidad:** Alta calidad, sin compresión excesiva
- **Tamaño de archivo:** 500KB - 5MB por imagen

### Condiciones de Captura

#### ✅ **Condiciones Ideales**
- **Iluminación:** Luz natural difusa, evitar sombras duras
- **Horario:** Mediodía con cielo parcialmente nublado
- **Distancia:** 1-3 metros del árbol/cultivo
- **Ángulo:** Múltiples ángulos (frontal, lateral, desde abajo)
- **Estabilidad:** Imágenes nítidas, sin movimiento

#### ⚠️ **Evitar**
- Contraluz excesivo
- Sombras muy marcadas
- Imágenes borrosas o con movimiento
- Resolución muy baja
- Compresión excesiva

### Composición de Imagen

#### 🎯 **Contenido Ideal por Imagen**
- **Avocados visibles:** 5-20 por imagen
- **Estados de madurez:** Variedad de estados en cada imagen
- **Elementos contextuales:** Hojas, ramas visibles
- **Fondo:** Natural del huerto/campo

#### 📏 **Tamaños de Avocados en Imagen**
- **Mínimo:** 30x30 píxeles por avocado
- **Recomendado:** 50x50 píxeles o más
- **Máximo:** Avocado puede ocupar hasta 30% de la imagen

## 🗂️ Archivo de Descripciones

Crear archivo `description/avocado_des.txt` con el siguiente contenido:

```
a dark green mature avocado ready for harvest, ripe
a light green unripe avocado still developing, unripe
an overripe dark avocado past optimal harvest time, overripe
a green avocado tree leaf, leaf
a brown avocado tree branch, branch
a small yellowish avocado tree flower, flower
soil or background or other elements, background
```

### Personalización de Descripciones

Puedes personalizar las descripciones según tu dataset específico:

```
# Para avocados Hass
a dark purple-black ripe Hass avocado with bumpy skin, ripe
a bright green unripe Hass avocado with smooth skin, unripe

# Para avocados Fuerte  
a dark green ripe Fuerte avocado with smooth skin, ripe
a light green unripe Fuerte avocado, unripe

# Contexto específico
a cluster of avocados hanging from branch, cluster
avocado tree canopy with multiple fruits, canopy
```

## 📊 Dataset de Ejemplo

### Distribución Recomendada de Imágenes

| Tipo de Imagen | Cantidad | Descripción |
|----------------|----------|-------------|
| **Árboles completos** | 30% | Vista general del árbol con múltiples avocados |
| **Secciones de ramas** | 40% | Ramas con 5-15 avocados visibles |
| **Primeros planos** | 20% | 2-5 avocados con detalle de madurez |
| **Vistas panorámicas** | 10% | Huerto/campo con múltiples árboles |

### Variabilidad Necesaria

#### 🌍 **Condiciones Ambientales**
- **Estaciones:** Diferentes épocas del año
- **Clima:** Soleado, nublado, después de lluvia
- **Hora del día:** Mañana, mediodía, tarde

#### 🌳 **Variedad de Árboles**
- **Edad:** Árboles jóvenes y maduros
- **Tamaño:** Diferentes alturas y densidades
- **Variedades:** Si aplica (Hass, Fuerte, etc.)

#### 🎨 **Variedad Visual**
- **Fondos:** Diferentes tipos de suelo, vegetación
- **Densidad:** Árboles con pocos/muchos frutos
- **Oclusión:** Avocados parcialmente ocultos por hojas

## 🛠️ Herramientas para Captura

### Equipos Recomendados
- **Cámara:** Smartphone moderno o cámara digital
- **Estabilización:** Trípode o estabilizador (opcional)
- **Iluminación:** Reflector para sombras (opcional)

### Apps Útiles
- **Android:** Open Camera, Camera FV-5
- **iOS:** Camera+ 2, ProCamera
- **Características útiles:** Grid, control manual de exposición

## 🔍 Control de Calidad

### Lista de Verificación Pre-Procesamiento

#### ✅ **Verificar cada imagen:**
- [ ] Resolución mínima cumplida (640x640)
- [ ] Imagen nítida y bien enfocada
- [ ] Al menos 3 avocados visibles
- [ ] Diferentes estados de madurez presentes
- [ ] Iluminación adecuada
- [ ] Formato correcto (JPG/PNG)

#### 🔍 **Verificar el dataset completo:**
- [ ] Distribución balanceada de estados de madurez
- [ ] Variedad de condiciones de captura
- [ ] Estructura de carpetas correcta
- [ ] Archivo de descripciones creado
- [ ] Total de imágenes suficiente (mínimo 100)

### Script de Validación

```python
# Verificar estructura del dataset
import os
from PIL import Image

def validate_avocado_dataset(dataset_path):
    """Valida estructura y calidad del dataset de avocados"""
    
    issues = []
    
    # Verificar cantidad mínima
    if total_images < 100:
        issues.append(f"Pocas imágenes: {total_images} (mínimo 100)")
    
    return issues, total_images

# Uso
issues, count = validate_avocado_dataset("./Images/avocado")
print(f"Total de imágenes: {count}")
if issues:
    print("Problemas encontrados:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ Dataset válido")
```

## 🌱 Ejemplos de Imágenes Ideales

### Imagen Tipo 1: Árbol Completo
```
Descripción: Vista completa de árbol de avocado
Contenido: 15-30 avocados visibles
Estados: Mezcla de maduros (40%), inmaduros (50%), sobre-maduros (10%)
Contexto: Hojas verdes, algunas ramas, fondo de huerto
Resolución: 1024x1024 o superior
```

### Imagen Tipo 2: Sección de Rama
```
Descripción: Rama principal con racimo de avocados
Contenido: 8-15 avocados agrupados
Estados: Mayoría del mismo estado de madurez
Contexto: Hojas circundantes, parte del tronco
Resolución: 800x800 mínimo
```

### Imagen Tipo 3: Primer Plano
```
Descripción: Detalle de 3-5 avocados
Contenido: Avocados claramente diferenciables
Estados: Diferentes estados para comparación
Contexto: Hojas como fondo natural
Resolución: 640x640 mínimo
```

## 📋 Lista de Verificación Final

### Antes del Procesamiento SDM-D

#### 📁 **Estructura de Archivos**
- [ ] Carpeta `Images/avocado/` creada
- [ ] Subcarpetas `train/`, `val/`, `test/` creadas
- [ ] Imágenes distribuidas correctamente
- [ ] Archivo `description/avocado_des.txt` creado

#### 📊 **Calidad del Dataset**
- [ ] Mínimo 100 imágenes total
- [ ] Distribución: 70% train, 15% val, 15% test
- [ ] Variedad de estados de madurez en cada subset
- [ ] Imágenes con resolución adecuada
- [ ] Calidad visual verificada

#### 🔧 **Configuración SDM-D**
- [ ] Checkpoint SAM2 descargado
- [ ] OpenCLIP instalado
- [ ] Descripción de avocados personalizada
- [ ] Carpeta de salida preparada

### Comando de Ejecución
```bash
# Procesamiento completo con análisis de avocados
python main_sdm_modular.py \
    --image_folder ./Images/avocado \
    --output_folder ./output/avocado \
    --description_file ./description/avocado_des.txt \
    --enable_visualizations \
    --box_visual \
    --color_visual \
    --save_json \
    --verbose

# Solo segmentación (para dataset muy grande)
python main_sdm_modular.py \
    --image_folder ./Images/avocado \
    --output_folder ./output/avocado \
    --only_segmentation \
    --enable_nms \
    --verbose
```

## 🎯 Optimizaciones Específicas para Avocados

### Parámetros Recomendados SAM2
```python
# Para avocados (objetos medianos-grandes)
points_per_side = 32          # Buena densidad para avocados
min_mask_region_area = 100    # Filtrar regiones muy pequeñas
enable_nms = True             # Importante para avocados agrupados
nms_threshold = 0.8           # Menos estricto para formas similares
```

### Descripciones Optimizadas
```
# Más específicas para mejor precisión
a dark green ripe avocado with mature coloring hanging from tree, ripe
a bright green immature avocado with firm appearance, unripe
a very dark overripe avocado with soft appearance, overripe
a typical avocado tree leaf with pointed shape, leaf
a brown woody avocado tree branch, branch
tiny yellowish avocado flower bud, flower
background soil dirt or sky elements, background
```

## 📈 Métricas Esperadas

### Resultados Típicos
- **Detección de avocados:** 85-95% de precisión
- **Clasificación de madurez:** 75-90% de precisión
- **Velocidad:** 2-5 segundos por imagen
- **Máscaras por imagen:** 10-25 (dependiendo de densidad)

### Factores que Afectan Precisión
- **Calidad de imagen:** +/- 10%
- **Variedad de dataset:** +/- 15%
- **Condiciones de iluminación:** +/- 5%
- **Densidad de avocados:** +/- 8%

## 🚀 Próximos Pasos

1. **Preparar Dataset**
   - Capturar/recopilar imágenes siguiendo esta guía
   - Organizar según estructura recomendada
   - Validar calidad usando script proporcionado

2. **Configurar Descripciones**
   - Crear archivo `avocado_des.txt`
   - Personalizar según variedades específicas
   - Probar con subset pequeño

3. **Ejecutar SDM-D**
   - Comenzar con procesamiento completo
   - Revisar resultados y ajustar parámetros
   - Generar análisis de cultivo

4. **Análisis de Resultados**
   - Revisar métricas de calidad
   - Generar reportes de cosecha
   - Ajustar descripciones si es necesario

## 🔗 Recursos Adicionales

### Datasets Públicos de Referencia
- **Avocado Dataset (Kaggle):** Para comparación de calidad
- **Fruit Detection Datasets:** Para técnicas de captura
- **Agricultural Vision:** Para mejores prácticas

### Herramientas Complementarias
- **LabelImg:** Para anotación manual si es necesaria
- **CVAT:** Para revisión de resultados
- **Agricultural Apps:** Para identificación de variedades

---

**💡 Consejo Final:** Comienza con un dataset pequeño (50-100 imágenes) para probar el pipeline completo antes de procesar datasets grandes. Esto te permitirá ajustar parámetros y descripciones eficientemente. estructura
    required_folders = ['train', 'val', 'test']
    for folder in required_folders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            issues.append(f"Falta carpeta: {folder}")
    
    # Verificar imágenes
    total_images = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_images += 1
                img_path = os.path.join(root, file)
                
                try:
                    img = Image.open(img_path)
                    width, height = img.size
                    
                    # Verificar resolución mínima
                    if width < 640 or height < 640:
                        issues.append(f"Resolución baja: {file} ({width}x{height})")
                    
                    # Verificar tamaño de archivo
                    file_size = os.path.getsize(img_path) / (1024*1024)  # MB
                    if file_size < 0.1:
                        issues.append(f"Archivo muy pequeño: {file} ({file_size:.2f}MB)")
                    elif file_size > 10:
                        issues.append(f"Archivo muy grande: {file} ({file_size:.2f}MB)")
                        
                except Exception as e:
                    issues.append(f"Error leyendo: {file} - {e}")
    
    # Verificar