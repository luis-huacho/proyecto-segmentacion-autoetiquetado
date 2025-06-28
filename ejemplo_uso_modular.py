#!/usr/bin/env python3
"""
Ejemplo de uso de los módulos SDM-D de forma independiente
Demuestra cómo usar cada componente por separado con enfoque en avocados
"""

import os
import cv2
import numpy as np
from pathlib import Path
import time

# Importar módulos
from segmentation import SAM2Segmentator
from annotations import CLIPAnnotator
from utiles import MaskProcessor, CLIPProcessor, LabelGenerator, FileManager, VisualizationManager
from utiles.logging_utils import SDMLogger, ProgressMonitor
from utiles.avocado_analytics import AvocadoAnalytics


def ejemplo_logging_sistema():
    """Ejemplo del sistema de logging"""
    print("🔹 Ejemplo 1: Sistema de Logging")
    print("-" * 50)

    # Configurar logger
    output_folder = "./ejemplos_output/logging"
    logger = SDMLogger(output_folder, enable_console=True)

    # Simular proceso de segmentación
    logger.start_phase("segmentation", 5)

    for i in range(5):
        start_time = time.time()

        # Simular procesamiento
        time.sleep(0.2)  # Simular trabajo
        processing_time = time.time() - start_time

        # Log del procesamiento
        logger.log_image_processing(
            f"avocado_{i:03d}.jpg",
            "segmentation",
            processing_time,
            masks_generated=np.random.randint(10, 25)
        )

        # Métricas específicas de segmentación
        logger.log_segmentation_metrics(
            f"avocado_{i:03d}.jpg",
            np.random.randint(20, 35),  # máscaras brutas
            np.random.randint(10, 25),  # máscaras filtradas
            True,  # NMS aplicado
            processing_time
        )

    logger.end_phase("segmentation")

    # Generar visualización de progreso
    progress_chart = logger.create_progress_visualization()
    print(f"📊 Gráfica de progreso: {progress_chart}")

    # Guardar reporte
    report_path = logger.save_session_report()
    print(f"📄 Reporte de sesión: {report_path}")


def ejemplo_monitor_progreso():
    """Ejemplo del monitor de progreso en tiempo real"""
    print("\n🔹 Ejemplo 2: Monitor de Progreso")
    print("-" * 50)

    # Configurar logger y monitor
    output_folder = "./ejemplos_output/monitor"
    logger = SDMLogger(output_folder, enable_console=True)
    monitor = ProgressMonitor(logger, update_interval=2)  # Actualizar cada 2 segundos

    # Iniciar monitoreo
    monitor.start_monitoring()

    # Simular procesamiento
    logger.start_phase("annotation", 8)

    for i in range(8):
        start_time = time.time()

        # Simular trabajo variable
        time.sleep(np.random.uniform(0.5, 1.5))
        processing_time = time.time() - start_time

        # Simular detecciones de avocados
        detections = np.random.randint(3, 12)

        logger.log_image_processing(
            f"field_avocado_{i:03d}.jpg",
            "annotation",
            processing_time,
            detections_made=detections
        )

        # Simular distribución de clases de avocados
        classifications = np.random.choice(['ripe', 'unripe', 'overripe'], detections)
        class_results = [{'label': cls} for cls in classifications]

        logger.log_annotation_metrics(
            f"field_avocado_{i:03d}.jpg",
            detections,
            class_results,
            processing_time
        )

    logger.end_phase("annotation")

    # Detener monitor
    monitor.stop_monitoring()

    print("✅ Monitor de progreso demostrado")


def ejemplo_analytics_avocados():
    """Ejemplo de analytics específicos para avocados"""
    print("\n🔹 Ejemplo 3: Analytics de Avocados")
    print("-" * 50)

    # Inicializar analytics
    output_folder = "./ejemplos_output/analytics"
    analytics = AvocadoAnalytics(output_folder)

    # Simular datos de detecciones de avocados
    analysis_results = []

    for i in range(10):
        # Simular detecciones variadas
        num_ripe = np.random.randint(5, 15)
        num_unripe = np.random.randint(8, 20)
        num_overripe = np.random.randint(0, 5)

        detections = []

        # Agregar avocados maduros
        for j in range(num_ripe):
            detections.append({
                'label': 'ripe',
                'xmin': np.random.randint(0, 500),
                'ymin': np.random.randint(0, 300),
                'xmax': np.random.randint(500, 800),
                'ymax': np.random.randint(300, 600)
            })

        # Agregar avocados inmaduros
        for j in range(num_unripe):
            detections.append({
                'label': 'unripe',
                'xmin': np.random.randint(0, 500),
                'ymin': np.random.randint(0, 300),
                'xmax': np.random.randint(500, 800),
                'ymax': np.random.randint(300, 600)
            })

        # Agregar avocados sobre-maduros
        for j in range(num_overripe):
            detections.append({
                'label': 'overripe',
                'xmin': np.random.randint(0, 500),
                'ymin': np.random.randint(0, 300),
                'xmax': np.random.randint(500, 800),
                'ymax': np.random.randint(300, 600)
            })

        # Crear análisis simulado
        analysis = analytics.analyze_avocado_detection(
            f"./ejemplos_output/sim_avocado_{i:03d}.jpg",  # Imagen simulada
            detections
        )

        if analysis:
            analysis_results.append(analysis)

        # Actualizar métricas globales
        analytics.add_processing_timeline_entry(time.time(), len(detections), f"field_{i:03d}.jpg")

    # Generar gráficas de analytics
    print("📊 Generando analytics de avocados...")

    # Gráfica de distribución de madurez
    maturity_chart = analytics.create_maturity_distribution_chart()
    if maturity_chart:
        print(f"   📊 Distribución de madurez: {maturity_chart}")

    # Análisis de tamaños
    size_chart = analytics.create_size_analysis_chart()
    if size_chart:
        print(f"   📊 Análisis de tamaños: {size_chart}")

    # Dashboard de calidad del cultivo
    dashboard = analytics.create_crop_quality_dashboard(analysis_results)
    if dashboard:
        print(f"   📊 Dashboard de calidad: {dashboard}")

    # Reporte completo
    report = analytics.export_analytics_report(analysis_results)
    print(f"   📄 Reporte de analytics: {report}")


def ejemplo_segmentacion_avocados():
    """Ejemplo de segmentación específica para avocados"""
    print("\n🔹 Ejemplo 4: Segmentación de Avocados")
    print("-" * 50)

    imagen_path = "./Images/avocado/field_001.jpg"  # Cambiar por tu imagen
    output_folder = "./ejemplos_output/segmentacion_avocados"

    if not os.path.exists(imagen_path):
        print(f"⚠️ Imagen no encontrada: {imagen_path}")
        print("   Usando configuración simulada...")
        return

    # Configurar logger
    logger = SDMLogger(output_folder, enable_console=True)

    # Inicializar segmentador con parámetros optimizados para avocados
    segmentador = SAM2Segmentator(
        checkpoint_path="./checkpoints/sam2_hiera_large.pt",
        points_per_side=32,  # Buena densidad para avocados
        min_mask_region_area=100,  # Filtrar regiones pequeñas
        logger=logger
    )

    print(f"🥑 Segmentando imagen de avocados: {imagen_path}")

    start_time = time.time()
    masks = segmentador.segment_image(imagen_path)
    processing_time = time.time() - start_time

    print(f"✅ Generadas {len(masks)} máscaras en {processing_time:.2f}s")

    # Aplicar NMS específico para avocados
    processor = MaskProcessor()
    filtered_masks = processor.apply_mask_nms(masks, threshold=0.8)  # Menos estricto para avocados

    print(f"🎯 Después de NMS: {len(filtered_masks)} máscaras")

    # Guardar resultados
    os.makedirs(output_folder, exist_ok=True)
    segmentador._save_masks(filtered_masks, output_folder)

    # Crear visualización
    visual_path = os.path.join(output_folder, "avocados_segmentados.png")
    segmentador._save_mask_visualization(filtered_masks, imagen_path, visual_path)

    print(f"📁 Resultados guardados en: {output_folder}")


def ejemplo_clasificacion_avocados():
    """Ejemplo de clasificación específica para avocados"""
    print("\n🔹 Ejemplo 5: Clasificación de Avocados")
    print("-" * 50)

    imagen_path = "./Images/avocado/field_001.jpg"
    descripciones_file = "./description/avocado_des.txt"

    if not os.path.exists(descripciones_file):
        print(f"⚠️ Archivo de descripciones no encontrado: {descripciones_file}")
        return

    # Configurar logger
    output_folder = "./ejemplos_output/clasificacion_avocados"
    logger = SDMLogger(output_folder, enable_console=True)

    # Inicializar anotador
    anotador = CLIPAnnotator(logger=logger)

    # Cargar descripciones de avocados
    texts, labels, label_dict = anotador.load_descriptions(descripciones_file)

    print(f"🥑 Clases de avocados detectables: {labels}")

    if not os.path.exists(imagen_path):
        print("⚠️ Imagen no encontrada, usando simulación...")

        # Simular clasificación
        for i, label in enumerate(['ripe', 'unripe', 'overripe']):
            print(f"   Simulando clasificación: {label}")

        return

    # Cargar imagen real
    imagen = cv2.imread(imagen_path)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Simular máscaras de avocados (en caso real vendrían de SAM2)
    altura, ancho = imagen_rgb.shape[:2]

    # Crear máscaras simuladas en diferentes posiciones
    masks_simuladas = []
    for i in range(3):
        mask = np.zeros((altura, ancho), dtype=np.uint8)
        x = np.random.randint(50, ancho - 100)
        y = np.random.randint(50, altura - 100)
        size = np.random.randint(40, 80)
        cv2.ellipse(mask, (x, y), (size, int(size * 1.3)), 0, 0, 360, 255, -1)
        masks_simuladas.append(mask)

    # Clasificar cada máscara
    resultados_clasificacion = []
    for i, mask in enumerate(masks_simuladas):
        etiqueta = anotador.classify_mask(imagen_rgb, mask, texts, labels)
        resultados_clasificacion.append(etiqueta)
        print(f"   Avocado {i + 1}: {etiqueta}")

    # Contar distribución
    from collections import Counter
    distribucion = Counter(resultados_clasificacion)
    print(f"📊 Distribución de madurez: {dict(distribucion)}")


def ejemplo_pipeline_completo_avocados():
    """Ejemplo de pipeline completo optimizado para avocados"""
    print("\n🔹 Ejemplo 6: Pipeline Completo para Avocados")
    print("-" * 50)

    imagen_path = "./Images/avocado/field_001.jpg"
    descripciones_file = "./description/avocado_des.txt"
    output_base = "./ejemplos_output/pipeline_avocados"

    # Verificar archivos
    archivos_necesarios = [imagen_path, descripciones_file]
    archivos_faltantes = [f for f in archivos_necesarios if not os.path.exists(f)]

    if archivos_faltantes:
        print("⚠️ Archivos faltantes para pipeline completo:")
        for archivo in archivos_faltantes:
            print(f"   - {archivo}")
        print("   Ejecutando versión simulada...")
        return ejemplo_pipeline_simulado_avocados(output_base)

    try:
        # Configurar logging completo
        logger = SDMLogger(output_base, enable_console=True)
        monitor = ProgressMonitor(logger, update_interval=5)

        # Inicializar analytics de avocados
        analytics = AvocadoAnalytics(output_base)

        print("🔄 Ejecutando pipeline completo para avocados...")

        # Iniciar monitoreo
        monitor.start_monitoring()

        # 1. Segmentación optimizada para avocados
        logger.start_phase("segmentation", 1)
        print("   1️⃣ Segmentación con parámetros optimizados para avocados...")

        segmentador = SAM2Segmentator(
            points_per_side=32,
            min_mask_region_area=100,
            logger=logger
        )
        masks = segmentador.segment_image(imagen_path)
        logger.end_phase("segmentation")

        # 2. Procesamiento de máscaras
        logger.start_phase("mask_processing", 1)
        print("   2️⃣ Aplicando NMS optimizado para avocados...")

        processor = MaskProcessor()
        masks_filtradas = processor.apply_mask_nms(masks, threshold=0.8)

        # Filtrar máscaras muy pequeñas (probablemente no son avocados)
        masks_filtradas = processor.filter_small_masks(masks_filtradas, min_area=150)
        logger.end_phase("mask_processing")

        # 3. Clasificación con CLIP
        logger.start_phase("annotation", 1)
        print("   3️⃣ Clasificando avocados por estado de madurez...")

        anotador = CLIPAnnotator(logger=logger)
        texts, labels, label_dict = anotador.load_descriptions(descripciones_file)

        imagen = cv2.imread(imagen_path)
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        detecciones = []
        for i, mask in enumerate(masks_filtradas):
            etiqueta = anotador.classify_mask(imagen_rgb, mask['segmentation'], texts, labels)

            # Solo procesar avocados (no hojas, ramas, etc.)
            if etiqueta in ['ripe', 'unripe', 'overripe']:
                coords = np.where(mask['segmentation'] > 0)
                if len(coords[0]) > 0:
                    ymin, ymax = coords[0].min(), coords[0].max()
                    xmin, xmax = coords[1].min(), coords[1].max()

                    detecciones.append({
                        'label': etiqueta,
                        'xmin': int(xmin), 'ymin': int(ymin),
                        'xmax': int(xmax), 'ymax': int(ymax)
                    })

        logger.end_phase("annotation")

        # 4. Analytics específicos de avocados
        print("   4️⃣ Generando analytics de avocados...")

        analysis = analytics.analyze_avocado_detection(imagen_path, detecciones)
        if analysis:
            print(f"      🥑 Avocados detectados: {analysis['avocado_count']}")
            print(f"      📊 Distribución: {dict(analysis['maturity_distribution'])}")
            print(
                f"      ⭐ Calidad del cultivo: {analysis.get('quality_indicators', {}).get('quality_category', 'N/A')}")

        # 5. Generar visualizaciones
        print("   5️⃣ Generando visualizaciones y reportes...")

        # Detener monitor
        monitor.stop_monitoring()

        # Guardar visualizaciones
        viz_manager = VisualizationManager()

        # Visualización con cajas de avocados
        box_visual_path = os.path.join(output_base, "avocados_detectados.png")
        viz_manager.create_box_visualization(imagen_path, detecciones, box_visual_path)

        # Gráficas de analytics
        maturity_chart = analytics.create_maturity_distribution_chart()
        dashboard = analytics.create_crop_quality_dashboard([analysis] if analysis else [])

        # Reporte final
        session_report = logger.save_session_report()
        analytics_report = analytics.export_analytics_report([analysis] if analysis else [])

        print(f"✅ Pipeline completo finalizado")
        print(f"   🥑 Avocados detectados: {len(detecciones)}")
        print(f"   📊 Visualizaciones: {box_visual_path}")
        print(f"   📊 Gráficas: {maturity_chart}")
        print(f"   📊 Dashboard: {dashboard}")
        print(f"   📄 Reportes: {session_report}, {analytics_report}")
        print(f"   📁 Todo en: {output_base}")

    except Exception as e:
        if 'monitor' in locals():
            monitor.stop_monitoring()
        print(f"❌ Error en pipeline: {e}")


def ejemplo_pipeline_simulado_avocados(output_base):
    """Pipeline simulado cuando no hay imágenes reales"""
    print("🔄 Ejecutando pipeline simulado de avocados...")

    # Configurar componentes
    logger = SDMLogger(output_base, enable_console=True)
    analytics = AvocadoAnalytics(output_base)

    # Simular detecciones de múltiples imágenes
    analysis_results = []

    for i in range(5):
        # Simular diferentes escenarios de cultivo
        detecciones_simuladas = []

        # Escenario variable de madurez
        if i < 2:  # Cultivo temprano - más inmaduros
            num_ripe, num_unripe, num_overripe = 3, 12, 1
        elif i < 4:  # Cultivo en punto - más maduros
            num_ripe, num_unripe, num_overripe = 10, 5, 2
        else:  # Cultivo tardío - algunos sobre-maduros
            num_ripe, num_unripe, num_overripe = 8, 3, 6

        # Crear detecciones simuladas
        for label, count in [('ripe', num_ripe), ('unripe', num_unripe), ('overripe', num_overripe)]:
            for j in range(count):
                detecciones_simuladas.append({
                    'label': label,
                    'xmin': np.random.randint(0, 600),
                    'ymin': np.random.randint(0, 400),
                    'xmax': np.random.randint(600, 1000),
                    'ymax': np.random.randint(400, 800)
                })

        # Analizar imagen simulada
        analysis = analytics.analyze_avocado_detection(
            f"sim_field_{i:03d}.jpg", detecciones_simuladas
        )

        if analysis:
            analysis_results.append(analysis)
            print(
                f"   Imagen {i + 1}: {analysis['avocado_count']} avocados, calidad {analysis.get('quality_indicators', {}).get('quality_category', 'N/A')}")

    # Generar reportes finales
    print("📊 Generando analytics finales...")

    maturity_chart = analytics.create_maturity_distribution_chart()
    size_chart = analytics.create_size_analysis_chart()
    dashboard = analytics.create_crop_quality_dashboard(analysis_results)
    report = analytics.export_analytics_report(analysis_results)

    print(f"✅ Pipeline simulado completado:")
    print(f"   📊 Gráfica de madurez: {maturity_chart}")
    print(f"   📊 Análisis de tamaños: {size_chart}")
    print(f"   📊 Dashboard: {dashboard}")
    print(f"   📄 Reporte: {report}")


def main():
    """Función principal que ejecuta todos los ejemplos"""
    print("🚀 SDM-D Framework - Ejemplos Especializados para Avocados")
    print("=" * 60)
    print("Este script demuestra el sistema de logging y analytics para avocados.")
    print("=" * 60)

    # Verificar dependencias básicas
    warnings = []
    if not os.path.exists("./checkpoints/sam2_hiera_large.pt"):
        warnings.append("Checkpoint SAM2 no encontrado")
    if not os.path.exists("./description/avocado_des.txt"):
        warnings.append("Archivo de descripciones de avocados no encontrado")

    if warnings:
        print("⚠️ Advertencias:")
        for warning in warnings:
            print(f"   • {warning}")
        print("   Algunos ejemplos usarán datos simulados")
        print()

    try:
        # Ejecutar ejemplos del sistema de logging
        ejemplo_logging_sistema()
        ejemplo_monitor_progreso()

        # Ejecutar ejemplos de analytics de avocados
        ejemplo_analytics_avocados()

        # Ejecutar ejemplos con datos reales si están disponibles
        if os.path.exists("./Images/avocado/"):
            ejemplo_segmentacion_avocados()
            ejemplo_clasificacion_avocados()
            ejemplo_pipeline_completo_avocados()
        else:
            print("\n⚠️ Carpeta ./Images/avocado/ no encontrada")
            print("   Ejecutando solo ejemplos simulados...")
            ejemplo_pipeline_simulado_avocados("./ejemplos_output/simulado")

    except Exception as e:
        print(f"\n❌ Error ejecutando ejemplos: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 Ejemplos completados!")
    print("📚 Revisa los archivos generados en ./ejemplos_output/")
    print("📊 Los logs y gráficas muestran el progreso y analytics en tiempo real.")


if __name__ == "__main__":
    main()
    stats = processor.get_mask_statistics(masks_filtradas)
    print("📊 Estadísticas de máscaras:")
    for key, value in stats.items():
        print(f"   {key}: {value}")


def ejemplo_clasificacion_clip():
    """Ejemplo de clasificación con CLIP"""
    print("\n🔹 Ejemplo 3: Clasificación con CLIP")
    print("-" * 50)

    imagen_path = "./Images/strawberry/12.png"
    descripciones_file = "./description/straw_des.txt"

    if not os.path.exists(imagen_path) or not os.path.exists(descripciones_file):
        print("⚠️ Archivos necesarios no encontrados")
        return

    # Inicializar anotador CLIP
    anotador = CLIPAnnotator()

    # Cargar descripciones
    texts, labels, label_dict = anotador.load_descriptions(descripciones_file)

    # Cargar imagen
    imagen = cv2.imread(imagen_path)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Simular una máscara (en caso real vendría de SAM2)
    altura, ancho = imagen_rgb.shape[:2]
    mask_simulada = np.zeros((altura, ancho), dtype=np.uint8)
    mask_simulada[50:150, 50:150] = 255  # Región cuadrada

    # Clasificar región
    print("🧠 Clasificando región con CLIP...")
    etiqueta_predicha = anotador.classify_mask(imagen_rgb, mask_simulada, texts, labels)
    print(f"   Etiqueta predicha: {etiqueta_predicha}")

    return etiqueta_predicha


def ejemplo_generacion_etiquetas():
    """Ejemplo de generación de etiquetas YOLO"""
    print("\n🔹 Ejemplo 4: Generación de etiquetas YOLO")
    print("-" * 50)

    # Simular máscara binaria
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 50, 255, -1)  # Círculo relleno

    # Inicializar generador
    generador = LabelGenerator()

    # Generar etiqueta YOLO bbox
    etiqueta_bbox = generador.generate_yolo_bbox_label(
        mask, class_id=0, img_width=200, img_height=200
    )
    print(f"📦 Etiqueta YOLO bbox: {etiqueta_bbox}")

    # Generar etiqueta YOLO polígono
    from scipy.ndimage import label as label_region
    labeled_mask, num_labels = label_region(mask)

    etiqueta_poligono = generador.generate_yolo_polygon_label(
        labeled_mask, class_id=0, img_width=200, img_height=200, num_labels=num_labels
    )
    print(f"🔺 Etiqueta YOLO polígono: {etiqueta_poligono[:100]}...")  # Mostrar solo inicio

    # Validar etiqueta
    es_valida, mensaje = generador.validate_yolo_label(etiqueta_bbox)
    print(f"✅ Validación bbox: {es_valida} - {mensaje}")


def ejemplo_manejo_archivos():
    """Ejemplo de manejo de archivos y estructura"""
    print("\n🔹 Ejemplo 5: Manejo de archivos")
    print("-" * 50)

    # Inicializar manejador
    manejador = FileManager("./ejemplos_output")

    # Crear estructura
    print("📁 Creando estructura de directorios...")
    manejador.create_output_structure()

    # Analizar dataset
    if os.path.exists("./Images/strawberry"):
        print("🔍 Analizando estructura del dataset...")
        estructura = manejador.organize_dataset_structure("./Images/strawberry")
        print(f"   Total de imágenes: {estructura['total_images']}")
        print(f"   Subcarpetas: {list(estructura['subfolders'].keys())}")

    # Obtener estadísticas (si existe output)
    stats = manejador.get_processing_stats("./ejemplos_output")
    print("📊 Estadísticas actuales:")
    for key, value in stats.items():
        print(f"   {key}: {value}")


def ejemplo_visualizaciones():
    """Ejemplo de generación de visualizaciones"""
    print("\n🔹 Ejemplo 6: Visualizaciones")
    print("-" * 50)

    # Inicializar manejador de visualización
    viz_manager = VisualizationManager()

    # Simular datos para visualización
    detecciones_simuladas = [
        {
            'label': 'ripe',
            'xmin': 50, 'ymin': 50,
            'xmax': 150, 'ymax': 150
        },
        {
            'label': 'unripe',
            'xmin': 200, 'ymin': 100,
            'xmax': 280, 'ymax': 180
        }
    ]

    # Crear imagen simulada
    imagen_simulada = np.ones((300, 400, 3), dtype=np.uint8) * 255
    cv2.circle(imagen_simulada, (100, 100), 40, (229, 76, 94), -1)  # Círculo rojo
    cv2.circle(imagen_simulada, (240, 140), 35, (146, 208, 80), -1)  # Círculo verde

    # Guardar imagen temporal
    temp_img_path = "./ejemplos_output/temp_imagen.png"
    os.makedirs(os.path.dirname(temp_img_path), exist_ok=True)
    cv2.imwrite(temp_img_path, imagen_simulada)

    # Generar visualización con cajas
    output_path = "./ejemplos_output/visualizacion_cajas.png"
    print(f"🎨 Generando visualización con cajas...")
    viz_manager.create_box_visualization(temp_img_path, detecciones_simuladas, output_path)
    print(f"   Guardado en: {output_path}")

    # Limpiar archivo temporal
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)


def ejemplo_pipeline_completo():
    """Ejemplo de pipeline completo usando todos los módulos"""
    print("\n🔹 Ejemplo 7: Pipeline completo")
    print("-" * 50)

    imagen_path = "./Images/strawberry/12.png"
    descripciones_file = "./description/straw_des.txt"
    output_base = "./ejemplos_output/pipeline_completo"

    if not os.path.exists(imagen_path) or not os.path.exists(descripciones_file):
        print("⚠️ Archivos necesarios no encontrados para pipeline completo")
        return

    try:
        print("🔄 Ejecutando pipeline completo...")

        # 1. Segmentación
        print("   1️⃣ Segmentación...")
        segmentador = SAM2Segmentator(points_per_side=16)
        masks = segmentador.segment_image(imagen_path)

        # 2. Procesamiento de máscaras
        print("   2️⃣ Procesamiento de máscaras...")
        processor = MaskProcessor()
        masks_filtradas = processor.apply_mask_nms(masks)

        # 3. Clasificación
        print("   3️⃣ Clasificación con CLIP...")
        anotador = CLIPAnnotator()
        texts, labels, label_dict = anotador.load_descriptions(descripciones_file)

        # 4. Generar etiquetas
        print("   4️⃣ Generando etiquetas...")
        imagen = cv2.imread(imagen_path)
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        altura, ancho = imagen_rgb.shape[:2]

        detecciones = []
        etiquetas_yolo = []

        for i, mask in enumerate(masks_filtradas[:3]):  # Solo primeras 3 para ejemplo
            # Clasificar
            etiqueta = anotador.classify_mask(imagen_rgb, mask['segmentation'], texts, labels)

            # Generar detección
            coords = np.where(mask['segmentation'] > 0)
            if len(coords[0]) > 0:
                ymin, ymax = coords[0].min(), coords[0].max()
                xmin, xmax = coords[1].min(), coords[1].max()

                detecciones.append({
                    'label': etiqueta,
                    'xmin': int(xmin), 'ymin': int(ymin),
                    'xmax': int(xmax), 'ymax': int(ymax)
                })

        # 5. Guardar resultados
        print("   5️⃣ Guardando resultados...")
        os.makedirs(output_base, exist_ok=True)

        # Guardar máscaras
        mask_folder = os.path.join(output_base, "masks")
        os.makedirs(mask_folder, exist_ok=True)
        processor.save_masks_as_images(masks_filtradas[:3], mask_folder)

        # Generar visualización
        viz_manager = VisualizationManager()
        viz_path = os.path.join(output_base, "resultado_final.png")
        viz_manager.create_box_visualization(imagen_path, detecciones, viz_path)

        print(f"✅ Pipeline completo finalizado")
        print(f"   📊 Máscaras procesadas: {len(masks_filtradas)}")
        print(f"   🏷️ Detecciones generadas: {len(detecciones)}")
        print(f"   📁 Resultados en: {output_base}")

    except Exception as e:
        print(f"❌ Error en pipeline: {e}")


def main():
    """Función principal que ejecuta todos los ejemplos"""
    print("🚀 SDM-D Framework - Ejemplos de Uso Modular")
    print("=" * 60)
    print("Este script demuestra cómo usar cada módulo de forma independiente.")
    print("=" * 60)

    # Verificar dependencias básicas
    if not os.path.exists("./checkpoints/sam2_hiera_large.pt"):
        print("⚠️ Advertencia: Checkpoint SAM2 no encontrado")
        print("   Algunos ejemplos pueden fallar")

    try:
        # Ejecutar ejemplos individuales
        ejemplo_procesamiento_mascaras()
        ejemplo_generacion_etiquetas()
        ejemplo_manejo_archivos()
        ejemplo_visualizaciones()

        # Ejemplos que requieren archivos específicos
        if os.path.exists("./Images/strawberry/12.png"):
            ejemplo_segmentacion_simple()
            ejemplo_clasificacion_clip()
            ejemplo_pipeline_completo()
        else:
            print("\n⚠️ Saltando ejemplos que requieren imágenes específicas")
            print("   Para ejecutar todos los ejemplos, asegúrate de tener:")
            print("   - ./Images/strawberry/12.png")
            print("   - ./description/straw_des.txt")

    except Exception as e:
        print(f"\n❌ Error ejecutando ejemplos: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 Ejemplos completados!")
    print("📚 Revisa el código fuente para entender cómo usar cada módulo.")


if __name__ == "__main__":
    main()