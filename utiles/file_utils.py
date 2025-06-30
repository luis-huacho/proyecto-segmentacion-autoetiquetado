"""
Utilidades para manejo de archivos y estructura de directorios
"""

import os
import json
import shutil
import cv2
from pathlib import Path
from datetime import datetime
from PIL import Image


class FileManager:
    """Clase para manejar operaciones con archivos y directorios"""

    def __init__(self, base_output_folder):
        """
        Inicializa el manejador de archivos

        Args:
            base_output_folder (str): Directorio base de salida
        """
        self.base_output_folder = base_output_folder

    def create_output_structure(self):
        """Crea la estructura completa de directorios para SDM"""
        folders = [
            'masks',
            'json',
            'labels',
            'visualizations',
            'metadata',
            'mask_idx_visual',
            'label_box_visual',
            'mask_color_visual'
        ]

        for folder in folders:
            folder_path = os.path.join(self.base_output_folder, folder)
            os.makedirs(folder_path, exist_ok=True)

        print(f"✅ Estructura de directorios creada en: {self.base_output_folder}")

    def get_output_structure_for_dataset(self):
        """
        Retorna la estructura de directorios para procesamiento de dataset

        Returns:
            dict: Diccionario con las rutas de salida organizadas
        """
        structure = {
            'masks': {
                'path': os.path.join(self.base_output_folder, 'masks'),
                'description': 'Máscaras de segmentación como imágenes PNG'
            },
            'visualizations': {
                'path': os.path.join(self.base_output_folder, 'visualizations'),
                'description': 'Visualizaciones de máscaras con índices'
            },
            'metadata': {
                'path': os.path.join(self.base_output_folder, 'metadata'),
                'description': 'Metadatos JSON de las máscaras'
            },
            'labels': {
                'path': os.path.join(self.base_output_folder, 'labels'),
                'description': 'Etiquetas YOLO generadas'
            },
            'json': {
                'path': os.path.join(self.base_output_folder, 'json'),
                'description': 'Archivos JSON con información detallada'
            }
        }

        # Crear directorios si no existen
        for folder_info in structure.values():
            os.makedirs(folder_info['path'], exist_ok=True)

        return structure

    def create_annotation_structure(self):
        """Crea estructura específica para anotaciones"""
        folders = [
            'labels',
            'label_box_visual',
            'mask_color_visual'
        ]

        for folder in folders:
            folder_path = os.path.join(self.base_output_folder, folder)
            os.makedirs(folder_path, exist_ok=True)

    def get_image_files(self, folder_path, extensions=None):
        """
        Obtiene lista de archivos de imagen en una carpeta

        Args:
            folder_path (str): Ruta de la carpeta
            extensions (list): Extensiones permitidas

        Returns:
            list: Lista de archivos de imagen
        """
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

        image_files = []

        if not os.path.exists(folder_path):
            return image_files

        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(file)

        return sorted(image_files)

    def organize_dataset_structure(self, dataset_folder):
        """
        Analiza y organiza la estructura del dataset

        Args:
            dataset_folder (str): Carpeta del dataset

        Returns:
            dict: Información de la estructura
        """
        structure_info = {
            'total_images': 0,
            'subfolders': {},
            'has_subfolders': False
        }

        if not os.path.exists(dataset_folder):
            return structure_info

        # Verificar si hay subcarpetas (train, val, test)
        items = os.listdir(dataset_folder)
        subfolders = [item for item in items if os.path.isdir(os.path.join(dataset_folder, item))]

        if subfolders:
            structure_info['has_subfolders'] = True

            for subfolder in subfolders:
                subfolder_path = os.path.join(dataset_folder, subfolder)
                image_files = self.get_image_files(subfolder_path)

                structure_info['subfolders'][subfolder] = {
                    'path': subfolder_path,
                    'image_count': len(image_files),
                    'image_files': image_files
                }
                structure_info['total_images'] += len(image_files)
        else:
            # No hay subcarpetas, todas las imágenes están en el directorio raíz
            image_files = self.get_image_files(dataset_folder)
            structure_info['subfolders']['root'] = {
                'path': dataset_folder,
                'image_count': len(image_files),
                'image_files': image_files
            }
            structure_info['total_images'] = len(image_files)

        return structure_info

    def save_json_metadata(self, data, output_path):
        """
        Guarda metadatos en formato JSON

        Args:
            data (dict): Datos a guardar
            output_path (str): Ruta de salida
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Convertir datos a formato serializable
            serializable_data = self._make_serializable(data)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error guardando JSON {output_path}: {e}")

    def load_json_metadata(self, file_path):
        """
        Carga metadatos desde archivo JSON

        Args:
            file_path (str): Ruta del archivo

        Returns:
            dict: Datos cargados
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error cargando JSON {file_path}: {e}")
            return {}

    def _make_serializable(self, obj):
        """
        Convierte objetos a formato serializable para JSON

        Args:
            obj: Objeto a convertir

        Returns:
            Objeto serializable
        """
        import numpy as np

        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.generic, np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif hasattr(obj, 'item'):  # Fallback para otros tipos numpy
            return obj.item()
        else:
            return obj

    def backup_existing_output(self, output_folder):
        """
        Crea backup del directorio de salida si existe

        Args:
            output_folder (str): Directorio de salida
        """
        if os.path.exists(output_folder):
            backup_folder = f"{output_folder}_backup"
            counter = 1

            # Encontrar nombre único para backup
            while os.path.exists(backup_folder):
                backup_folder = f"{output_folder}_backup_{counter}"
                counter += 1

            try:
                shutil.move(output_folder, backup_folder)
                print(f"📦 Backup creado: {backup_folder}")
            except Exception as e:
                print(f"⚠️ Error creando backup: {e}")

    def copy_dataset_structure(self, source_folder, target_folder, copy_images=False):
        """
        Copia la estructura de directorios de un dataset

        Args:
            source_folder (str): Carpeta fuente
            target_folder (str): Carpeta destino
            copy_images (bool): Si copiar también las imágenes
        """
        structure_info = self.organize_dataset_structure(source_folder)

        for subfolder_name, subfolder_info in structure_info['subfolders'].items():
            target_subfolder = os.path.join(target_folder, subfolder_name)
            os.makedirs(target_subfolder, exist_ok=True)

            if copy_images:
                source_subfolder = subfolder_info['path']
                for image_file in subfolder_info['image_files']:
                    source_path = os.path.join(source_subfolder, image_file)
                    target_path = os.path.join(target_subfolder, image_file)
                    shutil.copy2(source_path, target_path)

    def validate_dataset_integrity(self, dataset_folder, mask_folder=None):
        """
        Valida la integridad de un dataset

        Args:
            dataset_folder (str): Carpeta del dataset
            mask_folder (str): Carpeta de máscaras (opcional)

        Returns:
            dict: Reporte de validación
        """
        report = {
            'valid': True,
            'issues': [],
            'summary': {},
            'missing_masks': []
        }

        # Analizar estructura del dataset
        structure_info = self.organize_dataset_structure(dataset_folder)
        report['summary']['total_images'] = structure_info['total_images']
        report['summary']['subfolders'] = list(structure_info['subfolders'].keys())

        # Verificar imágenes corruptas
        corrupted_images = []
        for subfolder_name, subfolder_info in structure_info['subfolders'].items():
            for image_file in subfolder_info['image_files']:
                image_path = os.path.join(subfolder_info['path'], image_file)
                if not self._validate_image_file(image_path):
                    corrupted_images.append(image_path)

        if corrupted_images:
            report['valid'] = False
            report['issues'].append(f"Imágenes corruptas: {len(corrupted_images)}")
            report['corrupted_images'] = corrupted_images

        # Verificar máscaras si se proporciona carpeta
        if mask_folder and os.path.exists(mask_folder):
            for subfolder_name, subfolder_info in structure_info['subfolders'].items():
                for image_file in subfolder_info['image_files']:
                    image_name = Path(image_file).stem
                    mask_path = os.path.join(mask_folder, subfolder_name, image_name)

                    if not os.path.exists(mask_path):
                        report['missing_masks'].append(image_file)

        if report['missing_masks']:
            report['issues'].append(f"Máscaras faltantes: {len(report['missing_masks'])}")

        report['summary']['issues_count'] = len(report['issues'])

        return report

    def _validate_image_file(self, image_path):
        """
        Valida si un archivo de imagen es válido

        Args:
            image_path (str): Ruta de la imagen

        Returns:
            bool: True si la imagen es válida
        """
        try:
            # Intentar cargar con OpenCV
            img = cv2.imread(image_path)
            if img is None:
                return False

            # Verificar que tenga dimensiones válidas
            if img.shape[0] == 0 or img.shape[1] == 0:
                return False

            return True
        except Exception:
            try:
                # Intentar con PIL como backup
                with Image.open(image_path) as img:
                    img.verify()
                return True
            except Exception:
                return False

    def get_processing_stats(self, output_folder):
        """
        Obtiene estadísticas del procesamiento

        Args:
            output_folder (str): Carpeta de salida

        Returns:
            dict: Estadísticas del procesamiento
        """
        stats = {
            'masks_generated': 0,
            'labels_generated': 0,
            'visualizations_generated': 0,
            'json_files': 0,
            'total_files': 0
        }

        if not os.path.exists(output_folder):
            return stats

        # Contar máscaras
        masks_folder = os.path.join(output_folder, 'masks')
        if os.path.exists(masks_folder):
            for root, dirs, files in os.walk(masks_folder):
                stats['masks_generated'] += len([f for f in files if f.endswith('.png')])

        # Contar etiquetas
        labels_folder = os.path.join(output_folder, 'labels')
        if os.path.exists(labels_folder):
            for root, dirs, files in os.walk(labels_folder):
                stats['labels_generated'] += len([f for f in files if f.endswith('.txt')])

        # Contar visualizaciones
        visual_folders = ['visualizations', 'mask_idx_visual', 'label_box_visual', 'mask_color_visual']
        for visual_folder in visual_folders:
            folder_path = os.path.join(output_folder, visual_folder)
            if os.path.exists(folder_path):
                for root, dirs, files in os.walk(folder_path):
                    stats['visualizations_generated'] += len([f for f in files if f.endswith('.png')])

        # Contar archivos JSON
        json_folders = ['json', 'metadata']
        for json_folder in json_folders:
            folder_path = os.path.join(output_folder, json_folder)
            if os.path.exists(folder_path):
                for root, dirs, files in os.walk(folder_path):
                    stats['json_files'] += len([f for f in files if f.endswith('.json')])

        # Contar total de archivos
        for root, dirs, files in os.walk(output_folder):
            stats['total_files'] += len(files)

        return stats

    def generate_processing_report(self, output_folder, processing_time=None):
        """
        Genera reporte completo del procesamiento

        Args:
            output_folder (str): Carpeta de salida
            processing_time (float): Tiempo de procesamiento en segundos

        Returns:
            dict: Reporte completo
        """
        stats = self.get_processing_stats(output_folder)

        report = {
            'timestamp': self._get_timestamp(),
            'output_folder': output_folder,
            'processing_time': processing_time,
            'statistics': stats,
            'folder_structure': self._analyze_output_structure(output_folder)
        }

        return report

    def _get_timestamp(self):
        """Obtiene timestamp actual"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _analyze_output_structure(self, output_folder):
        """Analiza la estructura de la carpeta de salida"""
        structure = {}

        if os.path.exists(output_folder):
            for item in os.listdir(output_folder):
                item_path = os.path.join(output_folder, item)
                if os.path.isdir(item_path):
                    # Contar archivos en la subcarpeta
                    file_count = sum([len(files) for r, d, files in os.walk(item_path)])
                    structure[item] = {
                        'type': 'directory',
                        'file_count': file_count
                    }
                else:
                    structure[item] = {
                        'type': 'file',
                        'size': os.path.getsize(item_path)
                    }

        return structure

    def clean_empty_folders(self, root_folder):
        """
        Limpia carpetas vacías de manera recursiva

        Args:
            root_folder (str): Carpeta raíz para limpieza
        """
        for root, dirs, files in os.walk(root_folder, topdown=False):
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                try:
                    # Intentar eliminar si está vacía
                    os.rmdir(folder_path)
                    print(f"🗑️ Carpeta vacía eliminada: {folder_path}")
                except OSError:
                    # La carpeta no está vacía, continuar
                    pass

    def create_index_file(self, output_folder):
        """
        Crea archivo índice con información del contenido

        Args:
            output_folder (str): Carpeta de salida
        """
        index_content = {
            'created_at': self._get_timestamp(),
            'folder_structure': self._analyze_output_structure(output_folder),
            'statistics': self.get_processing_stats(output_folder),
            'description': 'Índice automático generado por SDM-D Framework'
        }

        index_path = os.path.join(output_folder, 'index.json')
        self.save_json_metadata(index_content, index_path)
        print(f"📋 Archivo índice creado: {index_path}")