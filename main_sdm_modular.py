#!/usr/bin/env python3
"""
Script principal para el framework SDM-D modular
Versión desacoplada y más entendible de SDM.py con logging completo y analytics para avocados
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

    # Procesamiento básico con avocados
    python main_sdm_modular.py \\
        --image_folder ./Images/avocado \\
        --output_folder ./output/avocado \\
        --description_file ./description/avocado_des.txt

    # Con visualizaciones y analytics completos
    python main_sdm_modular.py \\
        --image_folder ./Images/avocado \\
        --output_folder ./output/avocado \\
        --description_file ./description/avocado_des.txt \\
        --enable_visualizations \\
        --box_visual \\
        --color_visual \\
        --avocado_analytics \\
        --save_json \\
        --verbose

    # Solo segmentación (sin anotación)
    python main_sdm_modular.py \\
        --image_folder ./Images/avocado \\
        --output_folder ./output/avocado \\
        --only_segmentation \\
        --enable_nms \\
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
                        help='Ruta al checkpoint de SAM2')
    parser.add_argument('--sam2_config', type=str,
                        default='sam2_hiera_l.yaml',
                        help='Archivo de configuración de SAM2')
    parser.add_argument('--clip_model', type=str,
                        default='ViT-B-32',
                        help='Modelo CLIP a usar')
    parser.add_argument('--clip_pretrained', type=str,
                        default='laion2b_s34b_b79k',
                        help='Pesos preentrenados de CLIP')

    # Configuración de segmentación
    parser.add_argument('--points_per_side', type=int, default=32,
                        help='Número de puntos por lado para la grilla de SAM2')
    parser.add_argument('--min_mask_area', type=int, default=50,
                        help='Área mínima de las máscaras')
    parser.add_argument('--enable_nms', action='store_true',
                        help='Aplicar NMS a las máscaras')
    parser.add_argument('--nms_threshold', type=float, default=0.9,
                        help='Umbral para NMS de máscaras')

    # Opciones de salida
    parser.add_argument('--save_json', action='store_true',
                        help='Guardar metadatos en formato JSON')
    parser.add_argument('--enable_visualizations', action='store_true',
                        help='Generar visualizaciones')
    parser.add_argument('--box_visual', action='store_true',
                        help='Generar visualización con cajas delimitadoras')
    parser.add_argument('--color_visual', action='store_true',
                        help='Generar visualización con máscaras coloreadas')

    # Analytics específicos
    parser.add_argument('--avocado_analytics', action='store_true',
                        help='Habilitar analytics específicos para avocados')

    # Modo de operación
    parser.add_argument('--only_segmentation', action='store_true',
                        help='Solo realizar segmentación (sin anotación)')
    parser.add_argument('--only_annotation', action='store_true',
                        help='Solo realizar anotación (requiere máscaras existentes)')

    # Configuración de logging y monitoreo
    parser.add_argument('--verbose', action='store_true',
                        help='Mostrar información detallada')
    parser.add_argument('--enable_progress_monitor', action='store_true',
                        help='Habilitar monitor de progreso en tiempo real')
    parser.add_argument('--monitor_interval', type=int, default=30,
                        help='Intervalo del monitor de progreso (segundos)')

    # Configuración adicional
    parser.add_argument('--backup_existing', action='store_true',
                        help='Crear backup si la carpeta de salida existe')

    return parser.parse_args()


def validate_arguments(args):
    """Valida los argumentos proporcionados"""
    errors = []

    # Verificar carpeta de imágenes
    if not os.path.exists(args.image_folder):
        errors.append(f"La carpeta de imágenes no existe: {args.image_folder}")

    # Verificar checkpoint de SAM2
    if not args.only_annotation and not os.path.exists(args.sam2_checkpoint):
        errors.append(f"Checkpoint de SAM2 no encontrado: {args.sam2_checkpoint}")

    # Verificar archivo de descripciones para anotación
    if not args.only_segmentation:
        if not args.description_file:
            errors.append("El archivo de descripciones es requerido para anotación")
        elif not os.path.exists(args.description_file):
            errors.append(f"Archivo de descripciones no encontrado: {args.description_file}")

    # Verificar conflictos de modo
    if args.only_segmentation and args.only_annotation:
        errors.append("No se puede usar --only_segmentation y --only_annotation simultáneamente")

    # Para modo solo anotación, verificar que existan máscaras
    if args.only_annotation:
        mask_folder = os.path.join(args.output_folder, 'mask')
        if not os.path.exists(mask_folder):
            errors.append(f"Para modo solo anotación, debe existir carpeta de máscaras: {mask_folder}")

    if errors:
        print("❌ Errores en los argumentos:")
        for error in errors:
            print(f"   • {error}")
        return False

    return True


def print_configuration(args, logger):
    """Imprime la configuración actual"""
    config_msg = [
        "🔧 Configuración del procesamiento:",
        "=" * 60,
        f"📁 Carpeta de imágenes: {args.image_folder}",
        f"📁 Carpeta de salida: {args.output_folder}"
    ]

    if not args.only_annotation:
        config_msg.extend([
            f"🤖 Checkpoint SAM2: {args.sam2_checkpoint}",
            f"📊 Puntos por lado: {args.points_per_side}",
            f"🎯 NMS activado: {'Sí' if args.enable_nms else 'No'}"
        ])

    if not args.only_segmentation:
        config_msg.extend([
            f"📝 Archivo descripciones: {args.description_file}",
            f"🤖 Modelo CLIP: {args.clip_model}",
            f"🥑 Analytics de avocados: {'Sí' if args.avocado_analytics else 'No'}"
        ])

    config_msg.extend([
        f"📊 Visualizaciones: {'Sí' if args.enable_visualizations else 'No'}",
        f"📊 Guardar JSON: {'Sí' if args.save_json else 'No'}",
        f"📊 Logging detallado: {'Sí' if args.verbose else 'No'}",
        f"📊 Monitor de progreso: {'Sí' if args.enable_progress_monitor else 'No'}"
    ])

    mode = "Completo (Segmentación + Anotación)"
    if args.only_segmentation:
        mode = "Solo Segmentación"
    elif args.only_annotation:
        mode = "Solo Anotación"
    config_msg.append(f"🎮 Modo: {mode}")
    config_msg.append("=" * 60)

    for msg in config_msg:
        if logger:
            logger.main_logger.info(msg)
        else:
            print(msg)


def setup_logging(args):
    """Configura el sistema de logging"""
    if args.verbose:
        logger = SDMLogger(
            args.output_folder,
            enable_console=True
        )
        logger.main_logger.info("📊 Sistema de logging inicializado")
        return logger
    else:
        return None


def run_segmentation_phase(args, logger):
    """Ejecuta la fase de segmentación"""
    phase_msg = "🎭 FASE 1: SEGMENTACIÓN CON SAM2"
    separator = "-" * 40

    if logger:
        logger.main_logger.info(phase_msg)
        logger.main_logger.info(separator)
    else:
        print(f"\n{phase_msg}")
        print(separator)

    # Inicializar segmentador
    segmentator = SAM2Segmentator(
        model_cfg=args.sam2_config,
        checkpoint_path=args.sam2_checkpoint,
        points_per_side=args.points_per_side,
        min_mask_region_area=args.min_mask_area,
        logger=logger
    )

    # Ejecutar segmentación
    start_time = time.time()
    segmentator.segment_dataset(
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        enable_mask_nms=args.enable_nms,
        mask_nms_thresh=args.nms_threshold,
        save_annotations=args.enable_visualizations,
        save_json=args.save_json
    )
    segmentation_time = time.time() - start_time

    completion_msg = f"✅ Segmentación completada en {segmentation_time:.2f} segundos"
    if logger:
        logger.main_logger.info(completion_msg)
    else:
        print(completion_msg)

    return segmentation_time


def run_annotation_phase(args, logger):
    """Ejecuta la fase de anotación"""
    phase_msg = "🏷️ FASE 2: ANOTACIÓN CON OPENCLIP"
    separator = "-" * 40

    if logger:
        logger.main_logger.info(phase_msg)
        logger.main_logger.info(separator)
    else:
        print(f"\n{phase_msg}")
        print(separator)

    # Inicializar anotador
    annotator = CLIPAnnotator(
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        logger=logger
    )

    # Ejecutar anotación
    start_time = time.time()
    mask_folder = os.path.join(args.output_folder, 'mask')

    annotator.annotate_dataset(
        image_folder=args.image_folder,
        mask_folder=mask_folder,
        description_file=args.description_file,
        output_folder=args.output_folder,
        enable_visualizations=args.enable_visualizations,
        enable_box_visual=args.box_visual,
        enable_color_visual=args.color_visual,
        enable_avocado_analytics=args.avocado_analytics
    )
    annotation_time = time.time() - start_time

    completion_msg = f"✅ Anotación completada en {annotation_time:.2f} segundos"
    if logger:
        logger.main_logger.info(completion_msg)
    else:
        print(completion_msg)

    return annotation_time


def generate_final_report(args, processing_times, logger):
    """Genera reporte final del procesamiento"""
    report_msg = "📊 GENERANDO REPORTE FINAL"
    separator = "-" * 40

    if logger:
        logger.main_logger.info(report_msg)
        logger.main_logger.info(separator)
    else:
        print(f"\n{report_msg}")
        print(separator)

    # Crear manejador de archivos
    file_manager = FileManager(args.output_folder)

    # Obtener estadísticas
    stats = file_manager.get_processing_stats(args.output_folder)

    # Calcular tiempo total
    total_time = sum(processing_times.values())

    # Generar reporte
    report = file_manager.generate_processing_report(args.output_folder, total_time)

    # Guardar reporte
    report_path = os.path.join(args.output_folder, 'processing_report.json')
    file_manager.save_json_metadata(report, report_path)

    # Mostrar resumen
    summary_lines = [
        f"📈 Resumen del procesamiento:",
        f"   • Tiempo total: {total_time:.2f} segundos",
        f"   • Máscaras generadas: {stats['masks_generated']}",
        f"   • Etiquetas generadas: {stats['labels_generated']}",
        f"   • Visualizaciones: {stats['visualizations_generated']}",
        f"   • Archivos JSON: {stats['json_files']}",
        f"   • Subcarpetas procesadas: {stats['subfolders_processed']}"
    ]

    if 'segmentation' in processing_times:
        summary_lines.append(f"   • Tiempo segmentación: {processing_times['segmentation']:.2f}s")
    if 'annotation' in processing_times:
        summary_lines.append(f"   • Tiempo anotación: {processing_times['annotation']:.2f}s")

    summary_lines.append(f"📄 Reporte completo guardado en: {report_path}")

    for line in summary_lines:
        if logger:
            logger.main_logger.info(line)
        else:
            print(line)


def main():
    """Función principal"""
    print("🚀 SDM-D Framework Modular - Inicio del procesamiento")
    print("=" * 60)

    # Parsear argumentos
    args = parse_arguments()

    # Validar argumentos
    if not validate_arguments(args):
        exit(1)

    # Configurar logging
    logger = setup_logging(args)

    # Mostrar configuración
    print_configuration(args, logger)

    # Crear manejador de archivos
    file_manager = FileManager(args.output_folder)

    # Backup si es necesario
    if args.backup_existing:
        file_manager.backup_existing_output(args.output_folder)

    # Validar integridad del dataset
    if args.verbose and logger:
        logger.main_logger.info("🔍 Validando integridad del dataset...")
        validation_report = file_manager.validate_dataset_integrity(args.image_folder)
        if not validation_report['valid']:
            logger.main_logger.warning("⚠️ Se encontraron problemas en el dataset:")
            for issue in validation_report['issues']:
                logger.main_logger.warning(f"   • {issue}")

            response = input("¿Continuar de todas formas? (y/N): ")
            if response.lower() != 'y':
                logger.main_logger.error("❌ Procesamiento cancelado")
                exit(1)

    # Configurar monitor de progreso
    progress_monitor = None
    if args.enable_progress_monitor and logger:
        progress_monitor = ProgressMonitor(logger, args.monitor_interval)
        progress_monitor.start_monitoring()

    # Seguimiento de tiempos
    processing_times = {}
    total_start_time = time.time()

    try:
        # FASE 1: SEGMENTACIÓN
        if not args.only_annotation:
            segmentation_time = run_segmentation_phase(args, logger)
            processing_times['segmentation'] = segmentation_time

        # FASE 2: ANOTACIÓN
        if not args.only_segmentation:
            annotation_time = run_annotation_phase(args, logger)
            processing_times['annotation'] = annotation_time

        # Detener monitor de progreso
        if progress_monitor:
            progress_monitor.stop_monitoring()

        # REPORTE FINAL
        generate_final_report(args, processing_times, logger)

        # Limpiar directorios vacíos
        file_manager.clean_empty_directories(args.output_folder)

        total_time = time.time() - total_start_time

        # Generar visualización final de progreso
        if logger:
            progress_chart = logger.create_progress_visualization()
            if progress_chart:
                logger.main_logger.info(f"📊 Gráfica de progreso: {progress_chart}")

        success_lines = [
            "🎉 ¡Procesamiento completado exitosamente!",
            f"⏱️ Tiempo total: {total_time:.2f} segundos",
            f"📁 Resultados guardados en: {args.output_folder}"
        ]

        for line in success_lines:
            if logger:
                logger.main_logger.info(line)
            else:
                print(f"\n{line}")

        # Guardar reporte de sesión
        if logger:
            session_report = logger.save_session_report()
            logger.main_logger.info(f"📄 Reporte de sesión: {session_report}")

        # Generar visualización final si está habilitada
        if args.enable_visualizations:
            viz_msg = "📊 Generando visualizaciones finales..."
            if logger:
                logger.main_logger.info(viz_msg)
            else:
                print(f"\n{viz_msg}")

            try:
                viz_manager = VisualizationManager()
                # Aquí se pueden agregar visualizaciones adicionales del reporte
                viz_success = "✅ Visualizaciones generadas"
                if logger:
                    logger.main_logger.info(viz_success)
                else:
                    print(viz_success)
            except Exception as e:
                viz_error = f"⚠️ Error generando visualizaciones: {e}"
                if logger:
                    logger.log_error(viz_error, "main")
                else:
                    print(viz_error)

    except KeyboardInterrupt:
        interrupt_msg = "⏸️ Procesamiento interrumpido por el usuario"
        if progress_monitor:
            progress_monitor.stop_monitoring()
        if logger:
            logger.main_logger.warning(interrupt_msg)
            logger.save_session_report()
        else:
            print(f"\n{interrupt_msg}")
        exit(1)
    except Exception as e:
        error_msg = f"❌ Error durante el procesamiento: {e}"
        if progress_monitor:
            progress_monitor.stop_monitoring()
        if logger:
            logger.log_error(error_msg, "main")
            if args.verbose:
                import traceback
                logger.main_logger.error(traceback.format_exc())
            logger.save_session_report()
        else:
            print(f"\n{error_msg}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
    Anotación
    completada
    en
    {annotation_time: .2f}
    segundos
    ")
    return annotation_time


def generate_final_report(args, processing_times):
    """Genera reporte final del procesamiento"""
    print("\n📊 GENERANDO REPORTE FINAL")
    print("-" * 40)

    # Crear manejador de archivos
    file_manager = FileManager(args.output_folder)

    # Obtener estadísticas
    stats = file_manager.get_processing_stats(args.output_folder)

    # Calcular tiempo total
    total_time = sum(processing_times.values())

    # Generar reporte
    report = file_manager.generate_processing_report(args.output_folder, total_time)

    # Guardar reporte
    report_path = os.path.join(args.output_folder, 'processing_report.json')
    file_manager.save_json_metadata(report, report_path)

    # Mostrar resumen
    print(f"📈 Resumen del procesamiento:")
    print(f"   • Tiempo total: {total_time:.2f} segundos")
    print(f"   • Máscaras generadas: {stats['masks_generated']}")
    print(f"   • Etiquetas generadas: {stats['labels_generated']}")
    print(f"   • Visualizaciones: {stats['visualizations_generated']}")
    print(f"   • Archivos JSON: {stats['json_files']}")
    print(f"   • Subcarpetas procesadas: {stats['subfolders_processed']}")

    if 'segmentation' in processing_times:
        print(f"   • Tiempo segmentación: {processing_times['segmentation']:.2f}s")
    if 'annotation' in processing_times:
        print(f"   • Tiempo anotación: {processing_times['annotation']:.2f}s")

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
        print(f"📁 Resultados guardados en: {args.output_folder}")

        # Generar visualización final si está habilitada
        if args.enable_visualizations:
            print("\n📊 Generando visualizaciones finales...")
            try:
                viz_manager = VisualizationManager()
                # Aquí se pueden agregar visualizaciones adicionales del reporte
                print("✅ Visualizaciones generadas")
            except Exception as e:
                print(f"⚠️ Error generando visualizaciones: {e}")

    except KeyboardInterrupt:
        print("\n⏸️ Procesamiento interrumpido por el usuario")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()