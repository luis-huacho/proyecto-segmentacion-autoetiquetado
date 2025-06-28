#!/usr/bin/env python3
"""
Script principal para el framework SDM-D modular
Versión desacoplada y más entendible de SDM.py con logging completo y analytics para avocados
Compatible con Python 3.12 y dataset de clasificación de maduración 1-5
"""

import argparse
import time
import os
from pathlib import Path

from segmentation import SAM2Segmentator
from annotations import CLIPAnnotator
from utiles import FileManager, VisualizationManager
from utiles.logging_utils import SDMLogger, ProgressMonitor


def parse_arguments():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='SDM-D Framework Modular - Segmentación y Anotación sin etiquetas manuales',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

    # Procesamiento básico con avocados (5 clasificaciones de maduración)
    python main_sdm_modular.py \\
        --image_folder ./Images/avocado \\
        --output_folder ./output/avocado \\
        --description_file ./description/avocado_des.txt \\
        --avocado_ripening_dataset

    # Con visualizaciones y analytics completos para avocados
    python main_sdm_modular.py \\
        --image_folder ./Images/avocado \\
        --output_folder ./output/avocado \\
        --description_file ./description/avocado_des.txt \\
        --enable_visualizations \\
        --box_visual \\
        --color_visual \\
        --avocado_analytics \\
        --avocado_ripening_dataset \\
        --save_json \\
        --verbose

    # Solo segmentación optimizada para avocados
    python main_sdm_modular.py \\
        --image_folder ./Images/avocado \\
        --output_folder ./output/avocado \\
        --only_segmentation \\
        --enable_nms \\
        --points_per_side 32 \\
        --min_mask_area 100 \\
        --save_json \\
        --verbose
        """
    )

    # Argumentos requeridos
    parser.add_argument('--image_folder', type=str, required=True,
                        help='Carpeta con las imágenes a procesar')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Carpeta donde guardar los resultados')

    # Argumentos para anotación
    parser.add_argument('--description_file', type=str,
                        help='Archivo con descripciones para clasificación (requerido para anotación)')

    # Configuración de modelos
    parser.add_argument('--sam2_checkpoint', type=str,
                        default='./checkpoints/sam2_hiera_large.pt',
                        help='Ruta al checkpoint de SAM2 (default: sam2_hiera_large.pt)')
    parser.add_argument('--clip_model', type=str, default='ViT-B-32',
                        help='Modelo CLIP a usar (default: ViT-B-32)')
    parser.add_argument('--clip_pretrained', type=str, default='laion2b_s34b_b79k',
                        help='Pesos preentrenados de CLIP (default: laion2b_s34b_b79k)')

    # Configuración de segmentación SAM2
    parser.add_argument('--points_per_side', type=int, default=32,
                        help='Puntos por lado para SAM2 (default: 32, recomendado para avocados)')
    parser.add_argument('--min_mask_area', type=int, default=100,
                        help='Área mínima de máscara en píxeles (default: 100, optimizado para avocados)')
    parser.add_argument('--enable_nms', action='store_true',
                        help='Habilitar Non-Maximum Suppression para filtrar máscaras superpuestas')
    parser.add_argument('--nms_threshold', type=float, default=0.8,
                        help='Umbral de NMS (default: 0.8, menos estricto para avocados agrupados)')

    # Opciones de salida
    parser.add_argument('--save_json', action='store_true',
                        help='Guardar metadatos en formato JSON')
    parser.add_argument('--enable_visualizations', action='store_true',
                        help='Generar visualizaciones de resultados')
    parser.add_argument('--box_visual', action='store_true',
                        help='Crear visualización con cajas delimitadoras')
    parser.add_argument('--color_visual', action='store_true',
                        help='Crear visualización con máscaras coloreadas')

    # Modos de operación
    parser.add_argument('--only_segmentation', action='store_true',
                        help='Solo ejecutar segmentación (sin anotación)')
    parser.add_argument('--only_annotation', action='store_true',
                        help='Solo ejecutar anotación (requiere máscaras existentes)')
    parser.add_argument('--backup_existing', action='store_true',
                        help='Crear backup de resultados existentes antes de procesar')

    # Logging y monitoreo
    parser.add_argument('--verbose', action='store_true',
                        help='Mostrar información detallada del procesamiento')
    parser.add_argument('--enable_progress_monitor', action='store_true',
                        help='Habilitar monitor de progreso en tiempo real')
    parser.add_argument('--monitor_interval', type=int, default=30,
                        help='Intervalo de actualización del monitor en segundos (default: 30)')

    # Analytics específicos para avocados
    parser.add_argument('--avocado_analytics', action='store_true',
                        help='Habilitar analytics específicos para avocados (distribución de madurez, etc.)')
    parser.add_argument('--avocado_ripening_dataset', action='store_true',
                        help='Dataset con clasificación de maduración 1-5 (optimiza parámetros para esto)')

    return parser.parse_args()


def validate_arguments(args):
    """
    Valida argumentos de entrada

    Args:
        args: Argumentos parseados

    Returns:
        bool: True si los argumentos son válidos
    """
    # Validaciones básicas de rutas
    if not os.path.exists(args.image_folder):
        print(f"❌ Error: Carpeta de imágenes no encontrada: {args.image_folder}")
        return False

    # Para anotación se requiere archivo de descripción
    if not args.only_segmentation and not args.description_file:
        print("❌ Error: --description_file es requerido para anotación")
        print("   Usa --only_segmentation si solo quieres segmentar")
        return False

    if args.description_file and not os.path.exists(args.description_file):
        print(f"❌ Error: Archivo de descripción no encontrado: {args.description_file}")
        return False

    # Validar que no se usen modos conflictivos
    if args.only_segmentation and args.only_annotation:
        print("❌ Error: No se pueden usar --only_segmentation y --only_annotation juntos")
        return False

    # Validar checkpoint SAM2
    if not os.path.exists(args.sam2_checkpoint):
        print(f"❌ Error: Checkpoint SAM2 no encontrado: {args.sam2_checkpoint}")
        print("   Ejecuta: cd checkpoints && ./download_ckpts.sh")
        return False

    # Configuración específica para dataset de avocados con clasificación de maduración
    if args.avocado_ripening_dataset:
        # Ajustar parámetros automáticamente para este tipo de dataset
        if args.points_per_side == 32:  # Si es el valor por defecto
            args.points_per_side = 32  # Mantener, es óptimo para avocados
        if args.min_mask_area == 100:  # Si es el valor por defecto
            args.min_mask_area = 100  # Mantener, es óptimo para avocados
        if not args.enable_nms:
            args.enable_nms = True  # Activar automáticamente para avocados agrupados
            print("🥑 Dataset de avocados detectado: habilitando NMS automáticamente")
        if not args.avocado_analytics:
            args.avocado_analytics = True  # Activar analytics automáticamente
            print("🥑 Dataset de avocados detectado: habilitando analytics automáticamente")

    return True


def print_configuration(args):
    """Muestra la configuración actual"""
    print("\n🔧 CONFIGURACIÓN DEL PROCESAMIENTO")
    print("=" * 50)
    print(f"📁 Carpeta de imágenes: {args.image_folder}")
    print(f"📁 Carpeta de salida: {args.output_folder}")

    if args.description_file:
        print(f"📝 Archivo de descripciones: {args.description_file}")

    print(f"🤖 Checkpoint SAM2: {os.path.basename(args.sam2_checkpoint)}")
    print(f"🤖 Modelo CLIP: {args.clip_model}")

    print(f"🎯 Puntos por lado: {args.points_per_side}")
    print(f"📏 Área mínima de máscara: {args.min_mask_area}")
    print(f"🔄 NMS habilitado: {'✅' if args.enable_nms else '❌'}")

    if args.avocado_ripening_dataset:
        print("🥑 Modo: Dataset de avocados con clasificación de maduración (1-5)")
    if args.avocado_analytics:
        print("📊 Analytics de avocados: ✅")

    # Modo de operación
    if args.only_segmentation:
        print("🎭 Modo: Solo segmentación")
    elif args.only_annotation:
        print("🏷️ Modo: Solo anotación")
    else:
        print("🔄 Modo: Procesamiento completo (segmentación + anotación)")

    print("=" * 50)


def run_segmentation_phase(args):
    """Ejecuta la fase de segmentación"""
    print("\n🎭 FASE 1: SEGMENTACIÓN CON SAM2")
    print("-" * 40)

    start_time = time.time()

    # Crear segmentador con configuración específica para avocados
    segmentator = SAM2Segmentator(
        checkpoint_path=args.sam2_checkpoint,
        points_per_side=args.points_per_side,
        min_mask_region_area=args.min_mask_area
    )

    # Configurar NMS si está habilitado
    if args.enable_nms:
        segmentator.enable_nms(threshold=args.nms_threshold)

    # Ejecutar segmentación
    success = segmentator.segment_dataset(
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        save_json=args.save_json,
        verbose=args.verbose
    )

    if not success:
        raise RuntimeError("Error durante la segmentación")

    segmentation_time = time.time() - start_time
    print(f"✅ Segmentación completada en {segmentation_time:.2f} segundos")

    return segmentation_time


def run_annotation_phase(args):
    """Ejecuta la fase de anotación"""
    print("\n🏷️ FASE 2: ANOTACIÓN CON OPENCLIP")
    print("-" * 40)

    start_time = time.time()

    # Crear anotador
    annotator = CLIPAnnotator(
        clip_model=args.clip_model,
        pretrained=args.clip_pretrained
    )

    # Configurar analytics para avocados si está habilitado
    avocado_analytics = None
    if args.avocado_analytics:
        try:
            from utiles.avocado_analytics import AvocadoAnalytics
            avocado_analytics = AvocadoAnalytics(args.output_folder)
            annotator.set_avocado_analytics(avocado_analytics)
            print("🥑 Analytics de avocados habilitado")
        except ImportError:
            print("⚠️ No se pudo cargar AvocadoAnalytics, continuando sin analytics específicos")

    # Ejecutar anotación
    success = annotator.annotate_dataset(
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        description_file=args.description_file,
        enable_visualizations=args.enable_visualizations,
        enable_box_visual=args.box_visual,
        enable_color_visual=args.color_visual,
        save_json=args.save_json,
        verbose=args.verbose
    )

    if not success:
        raise RuntimeError("Error durante la anotación")

    annotation_time = time.time() - start_time
    print(f"✅ Anotación completada en {annotation_time:.2f} segundos")

    # Generar reporte de analytics para avocados si está habilitado
    if avocado_analytics:
        try:
            avocado_analytics.generate_comprehensive_report()
            print("📊 Reporte de analytics de avocados generado")
        except Exception as e:
            print(f"⚠️ Error generando reporte de analytics: {e}")

    return annotation_time


def generate_final_report(args, processing_times):
    """Genera reporte final del procesamiento"""
    print("\n📊 REPORTE FINAL")
    print("=" * 50)

    # Estadísticas de procesamiento
    total_time = sum(processing_times.values())
    print(f"⏱️ Tiempo total: {total_time:.2f} segundos")

    # Contar archivos generados
    output_path = Path(args.output_folder)
    stats = {
        'masks_generated': 0,
        'labels_generated': 0,
        'visualizations_generated': 0,
        'json_files': 0,
        'subfolders_processed': 0
    }

    # Contar máscaras
    mask_dir = output_path / 'mask'
    if mask_dir.exists():
        for subfolder in mask_dir.iterdir():
            if subfolder.is_dir():
                stats['subfolders_processed'] += 1
                stats['masks_generated'] += len(list(subfolder.glob('*.png')))

    # Contar etiquetas
    labels_dir = output_path / 'labels'
    if labels_dir.exists():
        for subfolder in labels_dir.iterdir():
            if subfolder.is_dir():
                stats['labels_generated'] += len(list(subfolder.glob('*.txt')))

    # Contar visualizaciones
    for viz_dir in ['mask_idx_visual', 'label_box_visual', 'mask_color_visual']:
        viz_path = output_path / viz_dir
        if viz_path.exists():
            for subfolder in viz_path.iterdir():
                if subfolder.is_dir():
                    stats['visualizations_generated'] += len(list(subfolder.glob('*.png')))

    # Contar JSONs
    stats['json_files'] = len(list(output_path.rglob('*.json')))

    # Mostrar estadísticas
    print(f"📈 ESTADÍSTICAS:")
    print(f"   • Máscaras generadas: {stats['masks_generated']}")
    print(f"   • Etiquetas generadas: {stats['labels_generated']}")
    print(f"   • Visualizaciones: {stats['visualizations_generated']}")
    print(f"   • Archivos JSON: {stats['json_files']}")
    print(f"   • Subcarpetas procesadas: {stats['subfolders_processed']}")

    if 'segmentation' in processing_times:
        print(f"   • Tiempo segmentación: {processing_times['segmentation']:.2f}s")
    if 'annotation' in processing_times:
        print(f"   • Tiempo anotación: {processing_times['annotation']:.2f}s")

    # Crear reporte JSON
    report_data = {
        'processing_summary': {
            'total_time_seconds': total_time,
            'processing_times': processing_times,
            'statistics': stats,
            'configuration': {
                'image_folder': args.image_folder,
                'output_folder': args.output_folder,
                'description_file': args.description_file,
                'sam2_checkpoint': args.sam2_checkpoint,
                'clip_model': args.clip_model,
                'points_per_side': args.points_per_side,
                'min_mask_area': args.min_mask_area,
                'enable_nms': args.enable_nms,
                'avocado_ripening_dataset': args.avocado_ripening_dataset,
                'avocado_analytics': args.avocado_analytics
            }
        }
    }

    import json
    report_path = output_path / 'processing_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"📄 Reporte completo guardado en: {report_path}")


def main():
    """Función principal"""
    print("🚀 SDM-D Framework Modular - Inicio del procesamiento")
    print("=" * 60)

    # Parsear argumentos
    args = parse_arguments()

    # Validar argumentos
    if not validate_arguments(args):
        exit(1)

    # Mostrar configuración
    print_configuration(args)

    # Crear manejador de archivos
    file_manager = FileManager(args.output_folder)

    # Backup si es necesario
    if args.backup_existing:
        file_manager.backup_existing_output(args.output_folder)

    # Validar integridad del dataset
    if args.verbose:
        print("\n🔍 Validando integridad del dataset...")
        validation_report = file_manager.validate_dataset_integrity(args.image_folder)
        if not validation_report['valid']:
            print("⚠️ Se encontraron problemas en el dataset:")
            for issue in validation_report['issues']:
                print(f"   • {issue}")

            response = input("¿Continuar de todas formas? (y/N): ")
            if response.lower() != 'y':
                print("❌ Procesamiento cancelado")
                exit(1)

    # Seguimiento de tiempos
    processing_times = {}
    total_start_time = time.time()

    try:
        # FASE 1: SEGMENTACIÓN
        if not args.only_annotation:
            segmentation_time = run_segmentation_phase(args)
            processing_times['segmentation'] = segmentation_time

        # FASE 2: ANOTACIÓN
        if not args.only_segmentation:
            annotation_time = run_annotation_phase(args)
            processing_times['annotation'] = annotation_time

        # REPORTE FINAL
        generate_final_report(args, processing_times)

        # Limpiar directorios vacíos
        file_manager.clean_empty_directories(args.output_folder)

        total_time = time.time() - total_start_time
        print(f"\n🎉 ¡Procesamiento completado exitosamente!")
        print(f"⏱️ Tiempo total: {total_time:.2f} segundos")

        if args.avocado_ripening_dataset:
            print("🥑 Dataset de avocados procesado con clasificación de maduración (1-5)")
            analytics_path = Path(args.output_folder) / 'analytics'
            if analytics_path.exists():
                print(f"📊 Revisa los analytics de avocados en: {analytics_path}")

    except KeyboardInterrupt:
        print("\n⚠️ Procesamiento interrumpido por el usuario")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()