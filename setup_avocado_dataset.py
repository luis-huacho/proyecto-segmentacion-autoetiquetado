#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de configuración para el dataset de avocados
Verifica dependencias y prepara el entorno de trabajo
"""

import sys
import os
import subprocess
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Verifica que se esté usando Python 3.12"""
    logger.info("🐍 Verificando versión de Python...")

    version = sys.version_info
    if version.major == 3 and version.minor == 12:
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        logger.error(f"❌ Python {version.major}.{version.minor}.{version.micro} detectado")
        logger.error("⚠️ Se requiere Python 3.12")
        return False


def check_required_files():
    """Verifica que los archivos principales del framework estén presentes"""
    logger.info("📁 Verificando archivos del framework...")

    required_files = [
        "main_sdm_modular.py",
        "process_avocado_dataset.py",
        "run_avocado_processing.py",
        "description/avocado_des.txt"
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        logger.error("❌ Archivos faltantes:")
        for file_path in missing_files:
            logger.error(f"   {file_path}")
        return False

    logger.info("✅ Todos los archivos del framework están presentes")
    return True


def check_dataset_structure(dataset_path):
    """Verifica la estructura del dataset de avocados"""
    logger.info("🥑 Verificando estructura del dataset...")

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        logger.error(f"❌ Dataset no encontrado: {dataset_path}")
        logger.info("💡 Asegúrate de que la ruta del dataset sea correcta")
        return False

    # Verificar archivos requeridos
    excel_file = dataset_path / "description.xlsx"
    images_folder = dataset_path / "images"

    if not excel_file.exists():
        logger.error(f"❌ Archivo Excel no encontrado: {excel_file}")
        return False

    if not images_folder.exists():
        logger.error(f"❌ Carpeta de imágenes no encontrada: {images_folder}")
        return False

    try:
        import pandas as pd
        df = pd.read_excel(excel_file)

        required_columns = ["File Name", "Ripening Index Classification"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"❌ Columnas faltantes en Excel: {missing_columns}")
            return False

        # Contar imágenes
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in images_folder.iterdir()
                       if f.suffix.lower() in image_extensions]

        logger.info(f"✅ Dataset válido: {len(df)} registros, {len(image_files)} imágenes")

        # Verificar clasificaciones
        classifications = df["Ripening Index Classification"].dropna().unique()
        logger.info(f"   Clasificaciones encontradas: {sorted(classifications)}")

    except Exception as e:
        logger.error(f"❌ Error leyendo Excel: {e}")
        return False

    return True


def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    logger.info("📦 Verificando dependencias...")

    # Mapeo correcto: nombre_pip -> nombre_importacion
    required_packages = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'opencv-python': 'cv2',  # CORREGIDO: opencv-python se importa como cv2
        'numpy': 'numpy',
        'pandas': 'pandas',
        'openpyxl': 'openpyxl',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'Pillow': 'PIL'  # CORREGIDO: Pillow se importa como PIL
    }

    missing_packages = []
    for pip_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            logger.info(f"   ✅ {pip_name}")
        except ImportError:
            missing_packages.append(pip_name)
            logger.error(f"   ❌ {pip_name}")

    if missing_packages:
        logger.error("❌ Paquetes faltantes:")
        for package in missing_packages:
            logger.error(f"   {package}")
        logger.info("💡 Instalar con: pip install " + " ".join(missing_packages))
        return False

    logger.info("✅ Todas las dependencias están instaladas")
    return True


def check_sam2_checkpoint():
    """Verifica que el checkpoint de SAM2 esté disponible"""
    logger.info("🤖 Verificando checkpoint SAM2...")

    checkpoint_path = Path("checkpoints/sam2_hiera_large.pt")

    if not checkpoint_path.exists():
        logger.error(f"❌ Checkpoint SAM2 no encontrado: {checkpoint_path}")
        logger.info("💡 Para descargar, ejecuta:")
        logger.info("   cd checkpoints && ./download_ckpts.sh")
        return False

    logger.info("✅ Checkpoint SAM2 encontrado")
    return True


def create_example_commands(dataset_path, output_path):
    """Crea archivo con comandos de ejemplo"""
    logger.info("📝 Creando comandos de ejemplo...")

    commands_content = f'''#!/bin/bash
# Comandos de ejemplo para procesar dataset de avocados
# Dataset: {dataset_path}
# Salida: {output_path}

echo "🥑 COMANDOS PARA PROCESAR DATASET DE AVOCADOS"
echo "=================================================="

echo "1️⃣ PREPARAR DATASET (crear splits train/val/test)"
python process_avocado_dataset.py \\
    --dataset_path "{dataset_path}" \\
    --output_path "{output_path}"

echo "2️⃣ PROCESAMIENTO COMPLETO CON ANALYTICS"
python main_sdm_modular.py \\
    --image_folder "{output_path}/prepared_dataset/Images/avocado" \\
    --output_folder "{output_path}/sdm_results" \\
    --description_file "{output_path}/prepared_dataset/description/avocado_des.txt" \\
    --avocado_ripening_dataset \\
    --enable_visualizations \\
    --box_visual \\
    --color_visual \\
    --save_json \\
    --verbose

echo "3️⃣ PIPELINE AUTOMÁTICO (todo en uno)"
python run_avocado_processing.py \\
    --dataset_path "{dataset_path}" \\
    --output_path "{output_path}"

echo "4️⃣ SOLO PREPARACIÓN DEL DATASET"
python run_avocado_processing.py \\
    --dataset_path "{dataset_path}" \\
    --output_path "{output_path}" \\
    --only_preparation

echo "5️⃣ SOLO SEGMENTACIÓN (sin clasificación)"
python main_sdm_modular.py \\
    --image_folder "{output_path}/prepared_dataset/Images/avocado" \\
    --output_folder "{output_path}/segmentation_only" \\
    --only_segmentation \\
    --enable_nms \\
    --verbose

echo "✅ Comandos listos para usar!"
'''

    commands_file = Path("avocado_commands.sh")
    with open(commands_file, 'w', encoding='utf-8') as f:
        f.write(commands_content)

    # Hacer ejecutable
    import stat
    commands_file.chmod(commands_file.stat().st_mode | stat.S_IEXEC)

    logger.info(f"✅ Comandos de ejemplo guardados en: {commands_file}")


def run_quick_test(dataset_path):
    """Ejecuta una prueba rápida del sistema"""
    logger.info("🧪 Ejecutando prueba rápida...")

    try:
        # Test de importación
        logger.info("   Probando importaciones...")

        # Test básicos de importación
        import torch
        import cv2
        import numpy as np
        import pandas as pd
        from PIL import Image

        logger.info("   ✅ Importaciones básicas exitosas")

        # Test de lectura del dataset si existe
        dataset_path = Path(dataset_path)
        if dataset_path.exists():
            logger.info("   Probando lectura del dataset...")
            excel_file = dataset_path / "description.xlsx"
            if excel_file.exists():
                df = pd.read_excel(excel_file)
                logger.info(f"   ✅ Dataset leído: {len(df)} registros")

        logger.info("🎉 Prueba rápida exitosa - Sistema listo!")
        return True

    except Exception as e:
        logger.error(f"❌ Error en prueba rápida: {e}")
        return False


def main():
    """Función principal de configuración"""
    print("🥑 CONFIGURACIÓN DEL DATASET DE AVOCADOS")
    print("=" * 50)

    # Solicitar rutas al usuario
    dataset_path = input("📁 Ruta al dataset (@avocado_dataset): ").strip()
    if not dataset_path:
        dataset_path = "./avocado_dataset"

    output_path = input("📁 Ruta de salida (./results_avocado): ").strip()
    if not output_path:
        output_path = "./results_avocado"

    print(f"\n🔍 Verificando configuración...")
    print(f"   Dataset: {dataset_path}")
    print(f"   Salida: {output_path}")

    # Lista de verificaciones
    checks = [
        ("Versión de Python", lambda: check_python_version()),
        ("Archivos del framework", lambda: check_required_files()),
        ("Estructura del dataset", lambda: check_dataset_structure(dataset_path)),
        ("Dependencias de Python", lambda: check_dependencies()),
        ("Checkpoint SAM2", lambda: check_sam2_checkpoint())
    ]

    all_passed = True

    print("\n🔧 VERIFICACIONES DEL SISTEMA")
    print("-" * 30)

    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            logger.error(f"❌ Error en {check_name}: {e}")
            all_passed = False

    print("\n" + "=" * 50)

    if all_passed:
        print("🎉 ¡CONFIGURACIÓN EXITOSA!")
        print("✅ Todos los componentes están listos")

        # Crear comandos de ejemplo
        create_example_commands(dataset_path, output_path)

        # Ejecutar prueba rápida
        if run_quick_test(dataset_path):
            print("\n🚀 PRÓXIMOS PASOS:")
            print("1. Revisa avocado_commands.sh para comandos de ejemplo")
            print("2. Ejecuta el pipeline automático:")
            print(f"   python run_avocado_processing.py --dataset_path '{dataset_path}' --output_path '{output_path}'")
            print("3. O ejecuta paso a paso siguiendo avocado_commands.sh")

    else:
        print("❌ CONFIGURACIÓN INCOMPLETA")
        print("⚠️ Soluciona los problemas indicados arriba antes de continuar")
        sys.exit(1)


if __name__ == "__main__":
    main()