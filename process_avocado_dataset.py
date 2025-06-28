#!/usr/bin/env python3
"""
Procesador especializado para el dataset de avocados con clasificación de índice de maduración
Compatible con Python 3.12
Archivo: process_avocado_dataset.py

Procesa el dataset @avocado_dataset/description.xlsx y las imágenes en @avocado_dataset/images/
para generar un dataset compatible con SDM-D modular
"""

import os
import sys
import pandas as pd
import shutil
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mapeo de índices de maduración a etiquetas descriptivas
RIPENING_INDEX_MAPPING = {
    1: ("underripe", "a light green underripe avocado with firm texture"),
    2: ("breaking", "a green avocado starting to ripen, breaking stage"),
    3: ("ripe_first", "a dark green ripe avocado in first stage, ready for harvest"),
    4: ("ripe_second", "a dark green to black ripe avocado in second stage, optimal harvest"),
    5: ("overripe", "a very dark overripe avocado past optimal harvest time")
}


class AvocadoDatasetProcessor:
    """
    Procesador para el dataset de avocados con clasificación de índice de maduración
    """

    def __init__(self, dataset_path: str, output_path: str):
        """
        Inicializa el procesador del dataset

        Args:
            dataset_path (str): Ruta al directorio @avocado_dataset
            output_path (str): Ruta donde crear el dataset procesado
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.images_path = self.dataset_path / "images"
        self.description_file = self.dataset_path / "description.xlsx"

        # Validar rutas
        self._validate_paths()

        # Crear directorios de salida
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _validate_paths(self):
        """Valida que las rutas requeridas existan"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"El directorio del dataset no existe: {self.dataset_path}")

        if not self.images_path.exists():
            raise FileNotFoundError(f"El directorio de imágenes no existe: {self.images_path}")

        if not self.description_file.exists():
            raise FileNotFoundError(f"El archivo de descripción no existe: {self.description_file}")

        logger.info(f"✅ Rutas validadas correctamente")
        logger.info(f"   📁 Dataset: {self.dataset_path}")
        logger.info(f"   🖼️ Imágenes: {self.images_path}")
        logger.info(f"   📊 Descripción: {self.description_file}")

    def load_dataset_description(self) -> pd.DataFrame:
        """
        Carga el archivo Excel con las descripciones del dataset

        Returns:
            pd.DataFrame: DataFrame con la información del dataset
        """
        try:
            df = pd.read_excel(self.description_file)
            logger.info(f"✅ Archivo Excel cargado: {len(df)} registros")

            # Validar columnas requeridas
            required_columns = ['File Name', 'Ripening Index Classification']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Columnas faltantes en el Excel: {missing_columns}")

            logger.info(f"📊 Columnas encontradas: {list(df.columns)}")

            # Mostrar distribución de clasificación
            classification_dist = df['Ripening Index Classification'].value_counts().sort_index()
            logger.info("📈 Distribución de clasificación de maduración:")
            for index, count in classification_dist.items():
                label, description = RIPENING_INDEX_MAPPING.get(index,
                                                                (f"unknown_{index}", f"unknown classification {index}"))
                logger.info(f"   {index} ({label}): {count} imágenes")

            return df

        except Exception as e:
            logger.error(f"❌ Error al cargar el archivo Excel: {e}")
            raise

    def verify_images_exist(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Verifica que las imágenes referenciadas en el Excel existan

        Args:
            df (pd.DataFrame): DataFrame con la información del dataset

        Returns:
            pd.DataFrame: DataFrame filtrado solo con imágenes existentes
        """
        logger.info("🔍 Verificando existencia de imágenes...")

        existing_images = []
        missing_images = []

        for _, row in df.iterrows():
            filename = row['File Name']
            # Asegurar extensión .jpg
            if not filename.lower().endswith('.jpg'):
                filename += '.jpg'

            image_path = self.images_path / filename

            if image_path.exists():
                existing_images.append(row)
            else:
                missing_images.append(filename)

        if missing_images:
            logger.warning(f"⚠️ Imágenes faltantes ({len(missing_images)}):")
            for img in missing_images[:10]:  # Mostrar solo las primeras 10
                logger.warning(f"   {img}")
            if len(missing_images) > 10:
                logger.warning(f"   ... y {len(missing_images) - 10} más")

        existing_df = pd.DataFrame(existing_images)
        logger.info(f"✅ Imágenes verificadas: {len(existing_df)}/{len(df)} encontradas")

        return existing_df

    def create_dataset_splits(self, df: pd.DataFrame, train_ratio: float = 0.7,
                              val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Divide el dataset en train/val/test manteniendo distribución balanceada

        Args:
            df (pd.DataFrame): DataFrame con la información del dataset
            train_ratio (float): Proporción para entrenamiento
            val_ratio (float): Proporción para validación
            test_ratio (float): Proporción para prueba

        Returns:
            Dict[str, pd.DataFrame]: Diccionario con los splits del dataset
        """
        logger.info("🔀 Creando splits del dataset...")

        # Verificar que las proporciones sumen 1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Las proporciones deben sumar 1.0")

        splits = {'train': [], 'val': [], 'test': []}

        # Dividir por cada clase para mantener distribución balanceada
        for ripening_class in df['Ripening Index Classification'].unique():
            class_data = df[df['Ripening Index Classification'] == ripening_class].copy()
            class_data = class_data.sample(frac=1).reset_index(drop=True)  # Shuffle

            n_total = len(class_data)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            n_test = n_total - n_train - n_val

            splits['train'].extend(class_data.iloc[:n_train].to_dict('records'))
            splits['val'].extend(class_data.iloc[n_train:n_train + n_val].to_dict('records'))
            splits['test'].extend(class_data.iloc[n_train + n_val:].to_dict('records'))

        # Convertir de vuelta a DataFrames
        result_splits = {}
        for split_name, split_data in splits.items():
            result_splits[split_name] = pd.DataFrame(split_data)
            logger.info(f"   {split_name}: {len(split_data)} imágenes")

        return result_splits

    def copy_images_to_splits(self, splits: Dict[str, pd.DataFrame]) -> None:
        """
        Copia las imágenes a los directorios correspondientes

        Args:
            splits (Dict[str, pd.DataFrame]): Splits del dataset
        """
        logger.info("📁 Copiando imágenes a directorios de splits...")

        for split_name, split_df in splits.items():
            split_dir = self.output_path / "Images" / "avocado" / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for _, row in split_df.iterrows():
                filename = row['File Name']
                if not filename.lower().endswith('.jpg'):
                    filename += '.jpg'

                src_path = self.images_path / filename
                dst_path = split_dir / filename

                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                else:
                    logger.warning(f"⚠️ Imagen no encontrada: {src_path}")

            logger.info(f"   ✅ {split_name}: {len(split_df)} imágenes copiadas")

    def create_description_file(self) -> None:
        """
        Crea el archivo de descripciones para SDM-D
        """
        logger.info("📝 Creando archivo de descripciones...")

        description_dir = self.output_path / "description"
        description_dir.mkdir(parents=True, exist_ok=True)

        description_file = description_dir / "avocado_des.txt"

        with open(description_file, 'w', encoding='utf-8') as f:
            f.write("# Descripciones para detección de avocados con clasificación de maduración\n")
            f.write("# Formato: descripción, etiqueta\n")
            f.write("# Generado automáticamente desde dataset clasificado\n\n")

            f.write("# Estados de madurez de avocados (basado en Ripening Index)\n")
            for index, (label, description) in RIPENING_INDEX_MAPPING.items():
                f.write(f"{description}, {label}\n")

            f.write("\n# Elementos adicionales del árbol de avocado\n")
            f.write("a green avocado tree leaf, leaf\n")
            f.write("a brown avocado tree branch, branch\n")
            f.write("a small yellowish avocado tree flower, flower\n")
            f.write("\n# Fondo y otros elementos\n")
            f.write("soil or background or other elements, background\n")

        logger.info(f"✅ Archivo de descripciones creado: {description_file}")

    def create_metadata_file(self, splits: Dict[str, pd.DataFrame]) -> None:
        """
        Crea archivo de metadatos del dataset procesado

        Args:
            splits (Dict[str, pd.DataFrame]): Splits del dataset
        """
        logger.info("📋 Creando archivo de metadatos...")

        metadata = {
            "dataset_info": {
                "name": "Avocado Ripening Classification Dataset",
                "source": str(self.dataset_path),
                "total_images": sum(len(df) for df in splits.values()),
                "classes": {index: label for index, (label, _) in RIPENING_INDEX_MAPPING.items()},
                "ripening_stages": {
                    1: "Underripe - Verde claro, firme",
                    2: "Breaking - Iniciando maduración",
                    3: "Ripe (First Stage) - Verde oscuro, listo para cosecha",
                    4: "Ripe (Second Stage) - Verde oscuro/negro, óptimo",
                    5: "Overripe - Muy oscuro, pasado del punto óptimo"
                }
            },
            "splits": {}
        }

        for split_name, split_df in splits.items():
            class_dist = split_df['Ripening Index Classification'].value_counts().sort_index()
            metadata["splits"][split_name] = {
                "total_images": len(split_df),
                "class_distribution": {int(k): int(v) for k, v in class_dist.items()}
            }

        import json
        metadata_file = self.output_path / "dataset_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Metadatos guardados: {metadata_file}")

    def create_processing_script(self) -> None:
        """
        Crea script de procesamiento para ejecutar SDM-D
        """
        logger.info("🔧 Creando script de procesamiento...")

        script_content = f'''#!/bin/bash
# Script de procesamiento para dataset de avocados
# Generado automáticamente por process_avocado_dataset.py

echo "🥑 Iniciando procesamiento del dataset de avocados..."

# Procesamiento completo con analytics de avocados
python main_sdm_modular.py \\
    --image_folder {self.output_path}/Images/avocado \\
    --output_folder {self.output_path}/output/avocado \\
    --description_file {self.output_path}/description/avocado_des.txt \\
    --enable_visualizations \\
    --box_visual \\
    --color_visual \\
    --avocado_analytics \\
    --enable_progress_monitor \\
    --save_json \\
    --verbose

echo "✅ Procesamiento completado!"
echo "📊 Revisa los resultados en: {self.output_path}/output/avocado"
'''

        script_file = self.output_path / "process_avocado.sh"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)

        # Hacer el script ejecutable en sistemas Unix
        import stat
        script_file.chmod(script_file.stat().st_mode | stat.S_IEXEC)

        logger.info(f"✅ Script de procesamiento creado: {script_file}")

    def process_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
                        test_ratio: float = 0.15) -> None:
        """
        Procesa el dataset completo

        Args:
            train_ratio (float): Proporción para entrenamiento
            val_ratio (float): Proporción para validación
            test_ratio (float): Proporción para prueba
        """
        logger.info("🚀 Iniciando procesamiento del dataset de avocados...")

        # 1. Cargar descripciones del Excel
        df = self.load_dataset_description()

        # 2. Verificar que las imágenes existan
        df = self.verify_images_exist(df)

        if len(df) == 0:
            raise ValueError("No se encontraron imágenes válidas en el dataset")

        # 3. Crear splits del dataset
        splits = self.create_dataset_splits(df, train_ratio, val_ratio, test_ratio)

        # 4. Copiar imágenes a los directorios correspondientes
        self.copy_images_to_splits(splits)

        # 5. Crear archivo de descripciones
        self.create_description_file()

        # 6. Crear archivo de metadatos
        self.create_metadata_file(splits)

        # 7. Crear script de procesamiento
        self.create_processing_script()

        logger.info("🎉 Dataset procesado exitosamente!")
        logger.info(f"📁 Dataset preparado en: {self.output_path}")
        logger.info("🔧 Para procesar con SDM-D, ejecuta:")
        logger.info(f"   bash {self.output_path}/process_avocado.sh")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Procesador para dataset de avocados con clasificación de maduración',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos de uso:

    # Procesamiento básico
    python process_avocado_dataset.py \\
        --dataset_path ./avocado_dataset \\
        --output_path ./prepared_avocado_dataset

    # Con splits personalizados
    python process_avocado_dataset.py \\
        --dataset_path ./avocado_dataset \\
        --output_path ./prepared_avocado_dataset \\
        --train_ratio 0.8 \\
        --val_ratio 0.1 \\
        --test_ratio 0.1
        '''
    )

    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Ruta al directorio @avocado_dataset')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Ruta donde crear el dataset procesado')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Proporción para entrenamiento (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Proporción para validación (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Proporción para prueba (default: 0.15)')

    args = parser.parse_args()

    try:
        processor = AvocadoDatasetProcessor(args.dataset_path, args.output_path)
        processor.process_dataset(args.train_ratio, args.val_ratio, args.test_ratio)

    except Exception as e:
        logger.error(f"❌ Error durante el procesamiento: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()