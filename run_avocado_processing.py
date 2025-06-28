#!/usr/bin/env python3
"""
Script de ejemplo para ejecutar el procesamiento del dataset de avocados
Compatible con Python 3.12
Archivo: run_avocado_processing.py

Este script ejecuta el pipeline completo:
1. Procesa el dataset original con clasificación
2. Ejecuta SDM-D con configuración optimizada para avocados
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command: str, description: str) -> bool:
    """
    Ejecuta un comando del sistema y maneja errores

    Args:
        command (str): Comando a ejecutar
        description (str): Descripción del comando

    Returns:
        bool: True si el comando fue exitoso
    """
    logger.info(f"🔄 {description}...")
    logger.info(f"   Comando: {command}")

    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completado exitosamente")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error en {description}:")
        logger.error(f"   Código de salida: {e.returncode}")
        logger.error(f"   STDOUT: {e.stdout}")
        logger.error(f"   STDERR: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Error inesperado en {description}: {e}")
        return False


def validate_paths(dataset_path: str, output_path: str) -> bool:
    """
    Valida que las rutas requeridas existan

    Args:
        dataset_path (str): Ruta al dataset original
        output_path (str): Ruta de salida

    Returns:
        bool: True si todas las rutas son válidas
    """
    dataset_path = Path(dataset_path)

    # Validar dataset original
    if not dataset_path.exists():
        logger.error(f"❌ Dataset no encontrado: {dataset_path}")
        return False

    required_files = [
        dataset_path / "description.xlsx",
        dataset_path / "images"
    ]

    for file_path in required_files:
        if not file_path.exists():
            logger.error(f"❌ Archivo/directorio requerido no encontrado: {file_path}")
            return False

    # Validar que existan algunas imágenes
    images_dir = dataset_path / "images"
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))

    if len(image_files) == 0:
        logger.error(f"❌ No se encontraron imágenes en: {images_dir}")
        return False

    logger.info(f"✅ Rutas validadas: {len(image_files)} imágenes encontradas")
    return True


def process_avocado_dataset_pipeline(dataset_path: str, output_path: str,
                                     skip_preparation: bool = False,
                                     only_preparation: bool = False) -> bool:
    """
    Ejecuta el pipeline completo de procesamiento

    Args:
        dataset_path (str): Ruta al dataset original
        output_path (str): Ruta de salida
        skip_preparation (bool): Saltar preparación del dataset
        only_preparation (bool): Solo preparar dataset, no ejecutar SDM-D

    Returns:
        bool: True si el procesamiento fue exitoso
    """
    logger.info("🥑 Iniciando pipeline de procesamiento de avocados")
    logger.info("=" * 60)

    # Validar rutas
    if not validate_paths(dataset_path, output_path):
        return False

    output_path = Path(output_path)
    prepared_dataset_path = output_path / "prepared_dataset"

    # PASO 1: Preparar dataset (si no se salta)
    if not skip_preparation:
        logger.info("📋 PASO 1: Preparando dataset...")

        command = f"""python process_avocado_dataset.py \\
            --dataset_path "{dataset_path}" \\
            --output_path "{prepared_dataset_path}" \\
            --train_ratio 0.7 \\
            --val_ratio 0.15 \\
            --test_ratio 0.15"""

        if not run_command(command, "Preparación del dataset"):
            return False

        # Verificar que se creó correctamente
        if not (prepared_dataset_path / "Images" / "avocado").exists():
            logger.error("❌ Error: El dataset preparado no se creó correctamente")
            return False

        logger.info(f"✅ Dataset preparado en: {prepared_dataset_path}")
    else:
        logger.info("⏭️ Saltando preparación del dataset (usando dataset existente)")

        # Verificar que el dataset preparado existe
        if not prepared_dataset_path.exists():
            logger.error(f"❌ Dataset preparado no encontrado: {prepared_dataset_path}")
            logger.error("   Ejecuta primero sin --skip_preparation")
            return False

    # Si solo queremos preparación, terminar aquí
    if only_preparation:
        logger.info("✅ Solo preparación solicitada - completado")
        return True

    # PASO 2: Ejecutar SDM-D con configuración optimizada
    logger.info("🔬 PASO 2: Ejecutando SDM-D para avocados...")

    sdm_output_path = output_path / "sdm_output"

    command = f"""python main_sdm_modular.py \\
        --image_folder "{prepared_dataset_path}/Images/avocado" \\
        --output_folder "{sdm_output_path}" \\
        --description_file "{prepared_dataset_path}/description/avocado_des.txt" \\
        --enable_visualizations \\
        --box_visual \\
        --color_visual \\
        --avocado_analytics \\
        --enable_progress_monitor \\
        --save_json \\
        --verbose \\
        --points_per_side 32 \\
        --min_mask_area 100 \\
        --enable_nms \\
        --nms_threshold 0.8"""

    if not run_command(command, "Procesamiento SDM-D"):
        return False

    # PASO 3: Generar reporte final
    logger.info("📊 PASO 3: Generando reporte final...")
    generate_final_summary(prepared_dataset_path, sdm_output_path, output_path)

    logger.info("🎉 Pipeline completado exitosamente!")
    logger.info(f"📁 Resultados finales en: {output_path}")

    return True


def generate_final_summary(prepared_path: Path, sdm_output_path: Path, output_path: Path):
    """
    Genera un resumen final del procesamiento

    Args:
        prepared_path (Path): Ruta del dataset preparado
        sdm_output_path (Path): Ruta de salida de SDM-D
        output_path (Path): Ruta de salida principal
    """
    try:
        summary_file = output_path / "processing_summary.txt"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("🥑 RESUMEN DEL PROCESAMIENTO DE AVOCADOS\n")
            f.write("=" * 50 + "\n\n")

            f.write("📁 ESTRUCTURA DE ARCHIVOS:\n")
            f.write(f"├── Dataset preparado: {prepared_path}\n")
            f.write(f"│   ├── Images/avocado/train/\n")
            f.write(f"│   ├── Images/avocado/val/\n")
            f.write(f"│   ├── Images/avocado/test/\n")
            f.write(f"│   └── description/avocado_des.txt\n")
            f.write(f"├── Resultados SDM-D: {sdm_output_path}\n")
            f.write(f"│   ├── mask/ (máscaras de segmentación)\n")
            f.write(f"│   ├── labels/ (etiquetas YOLO)\n")
            f.write(f"│   ├── mask_color_visual/ (visualizaciones)\n")
            f.write(f"│   ├── analytics/ (análisis de avocados)\n")
            f.write(f"│   └── logs/ (logs detallados)\n")
            f.write(f"└── Resumen: {summary_file}\n\n")

            f.write("🎯 CLASIFICACIÓN DE MADURACIÓN:\n")
            f.write("1 - Underripe: Verde claro, firme\n")
            f.write("2 - Breaking: Iniciando maduración\n")
            f.write("3 - Ripe (First Stage): Verde oscuro, listo para cosecha\n")
            f.write("4 - Ripe (Second Stage): Verde oscuro/negro, óptimo\n")
            f.write("5 - Overripe: Muy oscuro, pasado del punto óptimo\n\n")

            f.write("📊 ARCHIVOS GENERADOS:\n")
            f.write("- Máscaras de segmentación por imagen\n")
            f.write("- Etiquetas YOLO para entrenamiento\n")
            f.write("- Visualizaciones con cajas delimitadoras\n")
            f.write("- Análisis específico de avocados\n")
            f.write("- Reportes JSON con métricas\n")
            f.write("- Logs detallados del procesamiento\n\n")

            f.write("🔧 COMANDOS ÚTILES:\n")
            f.write("# Ver logs del procesamiento:\n")
            f.write(f"tail -f {sdm_output_path}/logs/main_*.log\n\n")
            f.write("# Ver análisis de avocados:\n")
            f.write(f"ls {sdm_output_path}/analytics/\n\n")
            f.write("# Reejecutar solo SDM-D:\n")
            f.write(
                f"python run_avocado_processing.py --dataset_path [DATASET] --output_path {output_path} --skip_preparation\n\n")

            f.write("✅ PROCESAMIENTO COMPLETADO EXITOSAMENTE\n")

        logger.info(f"📋 Resumen guardado en: {summary_file}")

    except Exception as e:
        logger.warning(f"⚠️ No se pudo generar el resumen: {e}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Pipeline completo para procesamiento de dataset de avocados',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos de uso:

    # Pipeline completo (preparación + SDM-D)
    python run_avocado_processing.py \\
        --dataset_path ./avocado_dataset \\
        --output_path ./results_avocado

    # Solo preparar dataset
    python run_avocado_processing.py \\
        --dataset_path ./avocado_dataset \\
        --output_path ./results_avocado \\
        --only_preparation

    # Solo ejecutar SDM-D (dataset ya preparado)
    python run_avocado_processing.py \\
        --dataset_path ./avocado_dataset \\
        --output_path ./results_avocado \\
        --skip_preparation

Estructura esperada del dataset:
    avocado_dataset/
    ├── description.xlsx    (con columnas: File Name, Ripening Index Classification)
    └── images/            (archivos .jpg referenciados en el Excel)

        '''
    )

    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Ruta al directorio del dataset original (@avocado_dataset)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Ruta donde guardar todos los resultados')
    parser.add_argument('--skip_preparation', action='store_true',
                        help='Saltar preparación del dataset (usar dataset ya preparado)')
    parser.add_argument('--only_preparation', action='store_true',
                        help='Solo preparar dataset, no ejecutar SDM-D')

    args = parser.parse_args()

    # Validar argumentos
    if args.skip_preparation and args.only_preparation:
        logger.error("❌ No se pueden usar --skip_preparation y --only_preparation juntos")
        sys.exit(1)

    try:
        success = process_avocado_dataset_pipeline(
            args.dataset_path,
            args.output_path,
            args.skip_preparation,
            args.only_preparation
        )

        if success:
            logger.info("🎉 Pipeline ejecutado exitosamente!")
            sys.exit(0)
        else:
            logger.error("❌ Pipeline falló")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("⚠️ Procesamiento interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()