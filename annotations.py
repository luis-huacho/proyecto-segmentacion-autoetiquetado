"""
Módulo de Anotación - SDM-D Framework
Maneja la clasificación de máscaras usando OpenCLIP y generación de etiquetas.
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import open_clip
from scipy.ndimage import label as label_region
import time

from utiles.clip_utils import CLIPProcessor
from utiles.label_utils import LabelGenerator
from utiles.file_utils import FileManager
from utiles.visualization_utils import VisualizationManager
from utiles.logging_utils import SDMLogger
from utiles.avocado_analytics import AvocadoAnalytics


class CLIPAnnotator:
    """
    Clase principal para manejar la anotación con OpenCLIP
    """

    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', logger=None):
        """
        Inicializa el anotador CLIP

        Args:
            model_name (str): Nombre del modelo CLIP
            pretrained (str): Nombre del modelo preentrenado
            logger (SDMLogger): Logger para seguimiento
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.logger = logger

        # Inicializar modelo
        self._init_clip_model()

        # Inicializar procesadores
        self.clip_processor = CLIPProcessor()
        self.label_generator = LabelGenerator()
        self.visualization_manager = VisualizationManager()

        # Inicializar analytics para avocados
        self.avocado_analytics = None

    def _init_clip_model(self):
        """Inicializa el modelo OpenCLIP"""
        try:
            # Configurar dispositivo
            if torch.cuda.is_available():
                self.device = 'cuda'
                torch.cuda.set_device(0)
            else:
                self.device = 'cpu'

            # Cargar modelo CLIP
            self.clip_model, _, self.clip_preprocessor = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained
            )

            # Mover a GPU si está disponible
            self.clip_model = self.clip_model.to(self.device)

            if self.logger:
                self.logger.main_logger.info(f"✅ OpenCLIP inicializado correctamente en {self.device}")
                self.logger.main_logger.info(f"📋 Modelo: {self.model_name} | Preentrenado: {self.pretrained}")
            else:
                print(f"✅ OpenCLIP inicializado correctamente en {self.device}")
                print(f"📋 Modelo: {self.model_name} | Preentrenado: {self.pretrained}")

        except Exception as e:
            error_msg = f"Error inicializando OpenCLIP: {e}"
            if self.logger:
                self.logger.log_error(error_msg, "annotation")
            else:
                print(f"❌ {error_msg}")
            raise

    def load_descriptions(self, description_file):
        """
        Carga descripciones desde archivo

        Args:
            description_file (str): Ruta al archivo de descripciones

        Returns:
            tuple: (texts, labels, label_dict)
        """
        texts = []
        labels = []
        label_dict = {}
        current_label = 0

        try:
            with open(description_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):  # Ignorar líneas vacías y comentarios
                        continue

                    # Formato: "descripción, etiqueta"
                    parts = line.split(',')
                    if len(parts) == 2:
                        text = parts[0].strip()
                        label = parts[1].strip()

                        # Agregar etiqueta al diccionario si no existe
                        if label not in label_dict:
                            label_dict[label] = current_label
                            current_label += 1

                        texts.append(text)
                        labels.append(label)
                    else:
                        warning_msg = f"Línea {line_num} malformada: {line}"
                        if self.logger:
                            self.logger.log_warning(warning_msg, "annotation")
                        else:
                            print(f"⚠️ {warning_msg}")

            if self.logger:
                self.logger.main_logger.info(f"📖 Cargadas {len(texts)} descripciones desde {description_file}")
                self.logger.main_logger.info(f"🏷️ Etiquetas: {list(label_dict.keys())}")
            else:
                print(f"📖 Cargadas {len(texts)} descripciones desde {description_file}")
                print(f"🏷️ Etiquetas: {list(label_dict.keys())}")

            return texts, labels, label_dict

        except Exception as e:
            error_msg = f"Error cargando descripciones: {e}"
            if self.logger:
                self.logger.log_error(error_msg, "annotation")
            else:
                print(f"❌ {error_msg}")
            raise

    def classify_mask(self, image, mask, texts, labels):
        """
        Clasifica una máscara individual usando CLIP

        Args:
            image (np.ndarray): Imagen RGB
            mask (np.ndarray): Máscara binaria
            texts (list): Lista de descripciones de texto
            labels (list): Lista de etiquetas correspondientes

        Returns:
            str: Etiqueta predicha
        """
        try:
            # Aplicar máscara y recortar objeto
            masked_image = self.clip_processor.apply_mask_to_image(image, mask)
            cropped_object = self.clip_processor.crop_object_from_background(masked_image)

            if cropped_object is None:
                return "unknown"

            # Preprocesar imagen para CLIP
            preprocessed_image = self.clip_preprocessor(cropped_object)
            image_input = torch.tensor(np.stack([preprocessed_image])).to(self.device)

            # Predecir con CLIP
            predicted_label = self.clip_processor.predict_with_clip(
                self.clip_model, image_input, texts, labels
            )

            return predicted_label

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error clasificando máscara: {e}", "annotation")
            else:
                print(f"❌ Error clasificando máscara: {e}")
            return "unknown"

    def set_avocado_analytics(self, avocado_analytics):
        """
        Configura analytics específicos para avocados

        Args:
            avocado_analytics (AvocadoAnalytics): Instancia de analytics
        """
        self.avocado_analytics = avocado_analytics

    def annotate_dataset(self, image_folder, output_folder, description_file,
                         mask_folder=None, enable_visualizations=True,
                         enable_box_visual=False, enable_color_visual=False,
                         enable_avocado_analytics=False, save_json=False,
                         verbose=False):
        """
        Anota un dataset completo de imágenes y máscaras

        Args:
            image_folder (str): Carpeta con imágenes
            output_folder (str): Carpeta de salida
            description_file (str): Archivo de descripciones
            mask_folder (str): Carpeta con máscaras (opcional, se inferirá si no se proporciona)
            enable_visualizations (bool): Generar visualizaciones
            enable_box_visual (bool): Generar visualización de cajas
            enable_color_visual (bool): Generar visualización coloreada
            enable_avocado_analytics (bool): Generar analytics de avocados
            save_json (bool): Guardar metadatos JSON
            verbose (bool): Logging detallado

        Returns:
            bool: True si la anotación fue exitosa
        """
        try:
            if self.logger:
                self.logger.main_logger.info("🏷️ Iniciando anotación del dataset...")
            else:
                print("🏷️ Iniciando anotación del dataset...")

            # Cargar descripciones
            texts, labels, label_dict = self.load_descriptions(description_file)

            # Crear estructura de salida
            file_manager = FileManager(output_folder)
            file_manager.create_annotation_structure()

            # Determinar carpeta de máscaras
            if mask_folder is None:
                # Inferir de la estructura del output de segmentación
                mask_folder = os.path.join(os.path.dirname(output_folder), 'masks')

            if not os.path.exists(mask_folder):
                error_msg = f"Carpeta de máscaras no encontrada: {mask_folder}"
                if self.logger:
                    self.logger.log_error(error_msg, "annotation")
                else:
                    print(f"❌ {error_msg}")
                return False

            # Analizar estructura del dataset
            dataset_structure = file_manager.organize_dataset_structure(image_folder)
            total_images = dataset_structure['total_images']

            if self.logger:
                self.logger.start_phase("annotation", total_images)

            processed_images = 0

            # Procesar cada subcarpeta
            for subfolder_name, subfolder_info in dataset_structure['subfolders'].items():
                if self.logger:
                    self.logger.main_logger.info(f"📂 Procesando subcarpeta: {subfolder_name}")

                subfolder_path = subfolder_info['path']

                # Verificar que la subcarpeta sea un directorio
                if not os.path.isdir(subfolder_path):
                    continue

                print(f"\n📂 Procesando subcarpeta: {subfolder_name}")

                # Obtener lista de imágenes
                image_files = [f for f in os.listdir(subfolder_path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                for image_file in image_files:
                    try:
                        image_name = Path(image_file).stem
                        image_path = os.path.join(subfolder_path, image_file)

                        # Determinar directorio de máscaras para esta imagen
                        mask_dir = os.path.join(mask_folder, subfolder_name, image_name)

                        print(f"  🖼️ Anotando: {image_file}")

                        # Verificar que existan máscaras
                        if not os.path.exists(mask_dir):
                            print(f"    ⚠️ No se encontraron máscaras para {image_file}")
                            continue

                        # Procesar imagen
                        results = self._process_single_image(
                            image_path, mask_dir, texts, labels, label_dict
                        )

                        if not results:
                            print(f"    ⚠️ No se procesaron máscaras para {image_file}")
                            continue

                        # Guardar etiquetas YOLO
                        label_output_path = os.path.join(output_folder, 'labels', subfolder_name, f'{image_name}.txt')
                        os.makedirs(os.path.dirname(label_output_path), exist_ok=True)
                        self._save_yolo_labels(results['yolo_labels'], label_output_path)

                        # Generar visualizaciones si está habilitado
                        if enable_visualizations:
                            self._generate_visualizations(
                                image_path, results, subfolder_name, image_file, output_folder,
                                enable_box_visual, enable_color_visual
                            )

                        # Guardar metadatos JSON si está habilitado
                        if save_json:
                            json_output_path = os.path.join(output_folder, 'json', subfolder_name, f'{image_name}.json')
                            os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
                            file_manager.save_json_metadata(results, json_output_path)

                        # Analytics de avocados si está habilitado
                        if enable_avocado_analytics and self.avocado_analytics:
                            self.avocado_analytics.process_image_results(image_file, results)

                        processed_images += 1
                        print(f"    ✅ Procesado: {len(results['detections'])} detecciones")

                        if self.logger:
                            self.logger.update_progress("annotation", processed_images)

                    except Exception as e:
                        print(f"    ❌ Error procesando {image_file}: {e}")
                        if self.logger:
                            self.logger.log_error(f"Error procesando {image_file}: {e}", "annotation")
                        continue

            # Finalizar fase
            if self.logger:
                self.logger.end_phase("annotation")
                self.logger.main_logger.info(f"📊 Resumen de anotación:")
                self.logger.main_logger.info(f"   • Imágenes procesadas: {processed_images}/{total_images}")

            print(f"\n✅ Anotación completada!")
            print(f"📊 Imágenes procesadas: {processed_images}/{total_images}")

            # Generar analytics de avocados si está habilitado
            if enable_avocado_analytics and self.avocado_analytics:
                self._generate_avocado_analytics()

            return True

        except Exception as e:
            error_msg = f"Error durante la anotación del dataset: {e}"
            if self.logger:
                self.logger.log_error(error_msg, "annotation")
            else:
                print(f"❌ {error_msg}")
            return False

    def _process_single_image(self, image_path, mask_dir, texts, labels, label_dict):
        """
        Procesa una imagen individual y sus máscaras

        Args:
            image_path (str): Ruta de la imagen
            mask_dir (str): Directorio con las máscaras
            texts (list): Textos de descripción
            labels (list): Etiquetas
            label_dict (dict): Diccionario de mapeo de etiquetas

        Returns:
            dict: Resultados del procesamiento
        """
        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                return None

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]

            # Obtener archivos de máscaras
            mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
            if not mask_files:
                return None

            detections = []
            yolo_labels = []
            annotated_masks = []

            # Mapeo explícito de etiquetas a IDs
            label_to_id = {label: i for i, label in enumerate(label_dict.keys())}

            # Procesar cada máscara
            for mask_file in mask_files:
                mask_path = os.path.join(mask_dir, mask_file)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if mask is None:
                    continue

                # Clasificar máscara
                predicted_label = self.classify_mask(image_rgb, mask, texts, labels)
                class_id = label_to_id.get(predicted_label, 0)

                # Generar bbox de la máscara
                coords = np.where(mask > 0)
                if len(coords[0]) == 0:
                    continue

                ymin, ymax = coords[0].min(), coords[0].max()
                xmin, xmax = coords[1].min(), coords[1].max()

                # Crear detección
                detection = {
                    'label': predicted_label,
                    'class_id': class_id,
                    'xmin': int(xmin), 'ymin': int(ymin),
                    'xmax': int(xmax), 'ymax': int(ymax),
                    'confidence': 1.0  # CLIP no proporciona score de confianza
                }
                detections.append(detection)

                # Generar etiqueta YOLO bbox
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height

                yolo_label = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                yolo_labels.append(yolo_label)

                # Agregar máscara anotada
                annotated_masks.append({
                    'mask': mask,
                    'label': predicted_label,
                    'bbox': (xmin, ymin, xmax, ymax)
                })

            return {
                'image_rgb': image_rgb,
                'detections': detections,
                'yolo_labels': yolo_labels,
                'annotated_masks': annotated_masks
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error procesando imagen {image_path}: {e}", "annotation")
            else:
                print(f"❌ Error procesando imagen {image_path}: {e}")
            return None

    def _save_yolo_labels(self, yolo_labels, output_path):
        """
        Guarda etiquetas en formato YOLO

        Args:
            yolo_labels (list): Lista de etiquetas YOLO
            output_path (str): Ruta de salida
        """
        try:
            with open(output_path, 'w') as f:
                for label in yolo_labels:
                    f.write(label + '\n')
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error guardando etiquetas YOLO: {e}", "annotation")
            else:
                print(f"❌ Error guardando etiquetas YOLO: {e}")

    def _generate_visualizations(self, image_path, results, subfolder, image_file, output_folder,
                                 enable_box_visual, enable_color_visual):
        """
        Genera visualizaciones de las detecciones

        Args:
            image_path (str): Ruta de la imagen
            results (dict): Resultados del procesamiento
            subfolder (str): Subcarpeta actual
            image_file (str): Nombre del archivo de imagen
            output_folder (str): Carpeta de salida
            enable_box_visual (bool): Generar visualización de cajas
            enable_color_visual (bool): Generar visualización coloreada
        """
        try:
            if enable_box_visual:
                # Visualización con cajas delimitadoras
                box_visual_dir = os.path.join(output_folder, 'label_box_visual', subfolder)
                os.makedirs(box_visual_dir, exist_ok=True)

                self.visualization_manager.create_box_visualization(
                    image_path, results['detections'],
                    os.path.join(box_visual_dir, image_file)
                )

            if enable_color_visual:
                # Visualización con máscaras coloreadas
                color_visual_dir = os.path.join(output_folder, 'mask_color_visual', subfolder)
                os.makedirs(color_visual_dir, exist_ok=True)

                self.visualization_manager.create_color_mask_visualization(
                    results['image_rgb'], results['annotated_masks'], results['detections'],
                    os.path.join(color_visual_dir, image_file)
                )

        except Exception as e:
            print(f"    ⚠️ Error generando visualizaciones: {e}")

    def _generate_avocado_analytics(self):
        """Genera analytics específicos para avocados"""
        if not self.avocado_analytics:
            return

        try:
            if self.logger:
                self.logger.main_logger.info("📊 Generando analytics de avocados...")
            else:
                print("📊 Generando analytics de avocados...")

            # Crear gráficas de distribución de madurez
            chart_path = self.avocado_analytics.create_maturity_distribution_chart()
            if chart_path and self.logger:
                self.logger.main_logger.info(f"📊 Gráfica de madurez: {chart_path}")

            # Crear análisis de tamaños
            size_chart_path = self.avocado_analytics.create_size_analysis_chart()
            if size_chart_path and self.logger:
                self.logger.main_logger.info(f"📊 Análisis de tamaños: {size_chart_path}")

            # Exportar reporte completo
            report_path = self.avocado_analytics.export_analytics_report()
            if self.logger:
                self.logger.main_logger.info(f"📄 Reporte de analytics: {report_path}")

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error generando analytics de avocados: {e}", "annotation")
            else:
                print(f"❌ Error generando analytics de avocados: {e}")


def main():
    """Función principal para testing del módulo"""
    import argparse

    parser = argparse.ArgumentParser(description='Módulo de Anotación OpenCLIP')
    parser.add_argument('--image_folder', type=str, required=True, help='Carpeta con imágenes')
    parser.add_argument('--mask_folder', type=str, required=True, help='Carpeta con máscaras')
    parser.add_argument('--description_file', type=str, required=True, help='Archivo de descripciones')
    parser.add_argument('--output_folder', type=str, required=True, help='Carpeta de salida')
    parser.add_argument('--box_visual', action='store_true', help='Generar visualización de cajas')
    parser.add_argument('--color_visual', action='store_true', help='Generar visualización coloreada')
    parser.add_argument('--avocado_analytics', action='store_true', help='Habilitar analytics de avocados')
    parser.add_argument('--verbose', action='store_true', help='Logging detallado')

    args = parser.parse_args()

    # Configurar logger si está en modo verbose
    logger = None
    if args.verbose:
        logger = SDMLogger(args.output_folder, enable_console=True)

    # Crear anotador
    annotator = CLIPAnnotator(logger=logger)

    # Configurar analytics de avocados si está habilitado
    if args.avocado_analytics:
        try:
            avocado_analytics = AvocadoAnalytics(args.output_folder)
            annotator.set_avocado_analytics(avocado_analytics)
        except ImportError:
            print("⚠️ Analytics de avocados no disponibles")

    # Anotar dataset
    success = annotator.annotate_dataset(
        image_folder=args.image_folder,
        mask_folder=args.mask_folder,
        description_file=args.description_file,
        output_folder=args.output_folder,
        enable_visualizations=True,
        enable_box_visual=args.box_visual,
        enable_color_visual=args.color_visual,
        enable_avocado_analytics=args.avocado_analytics
    )

    # Guardar reporte si hay logger
    if logger:
        logger.save_session_report()

    return success


if __name__ == "__main__":
    main()