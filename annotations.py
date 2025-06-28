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
                self.logger.log_error(error_msg, "annotation", context=description_file)
            else:
                print(f"❌ {error_msg}")
            raise

    def classify_mask(self, image, mask, texts, labels):
        """
        Clasifica una máscara individual usando CLIP

        Args:
            image (np.ndarray): Imagen original
            mask (np.ndarray): Máscara binaria
            texts (list): Lista de descripciones
            labels (list): Lista de etiquetas

        Returns:
            str: Etiqueta predicha
        """
        try:
            # Aplicar máscara a la imagen
            masked_image = self.clip_processor.apply_mask_to_image(image, mask)

            # Recortar región de interés
            cropped_image = self.clip_processor.crop_object_from_background(masked_image)

            # Validar calidad de imagen
            if not self.clip_processor.validate_image_quality(cropped_image):
                if self.logger:
                    self.logger.log_warning("Imagen de baja calidad para clasificación", "annotation")
                return "others"

            # Preprocesar para CLIP
            image_preprocessed = self.clip_preprocessor(cropped_image)
            image_input = torch.tensor(np.stack([image_preprocessed])).to(self.device)

            # Predecir con CLIP
            predicted_label = self.clip_processor.predict_with_clip(
                self.clip_model, image_input, texts, labels
            )

            return predicted_label

        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error clasificando máscara: {e}", "annotation")
            else:
                print(f"⚠️ Error clasificando máscara: {e}")
            return "others"  # Etiqueta por defecto

    def annotate_dataset(self, image_folder, mask_folder, description_file, output_folder,
                         enable_visualizations=False, enable_box_visual=False, enable_color_visual=False,
                         enable_avocado_analytics=True):
        """
        Anota un dataset completo

        Args:
            image_folder (str): Carpeta con imágenes originales
            mask_folder (str): Carpeta con máscaras generadas
            description_file (str): Archivo con descripciones
            output_folder (str): Carpeta de salida
            enable_visualizations (bool): Generar visualizaciones
            enable_box_visual (bool): Visualizar cajas delimitadoras
            enable_color_visual (bool): Visualizar máscaras coloreadas
            enable_avocado_analytics (bool): Habilitar análisis específico de avocados
        """

        # Cargar descripciones
        texts, labels, label_dict = self.load_descriptions(description_file)

        # Detectar si es dataset de avocados
        is_avocado_dataset = any(label in ['ripe', 'unripe', 'overripe'] for label in labels)

        # Inicializar analytics de avocados si aplica
        if enable_avocado_analytics and is_avocado_dataset:
            self.avocado_analytics = AvocadoAnalytics(output_folder)
            if self.logger:
                self.logger.main_logger.info("🥑 Analytics de avocados habilitado")

        # Crear estructura de carpetas
        file_manager = FileManager(output_folder)
        file_manager.create_annotation_structure()

        # Analizar dataset
        dataset_structure = file_manager.organize_dataset_structure(image_folder)
        total_images = dataset_structure['total_images']

        if self.logger:
            self.logger.start_phase("annotation", total_images)
            self.logger.main_logger.info(f"📁 Carpetas a procesar: {list(dataset_structure['subfolders'].keys())}")

        processed_images = 0
        analysis_results = []  # Para analytics de avocados

        # Procesar cada subcarpeta
        for subfolder_name, subfolder_info in dataset_structure['subfolders'].items():
            if self.logger:
                self.logger.main_logger.info(f"📂 Procesando subcarpeta: {subfolder_name}")

            for image_file in subfolder_info['image_files']:
                try:
                    image_name = Path(image_file).stem
                    image_path = os.path.join(subfolder_info['path'], image_file)
                    mask_dir = os.path.join(mask_folder, subfolder_name, image_name)

                    # Verificar que existan máscaras
                    if not os.path.exists(mask_dir):
                        warning_msg = f"No se encontraron máscaras para {image_file}"
                        if self.logger:
                            self.logger.log_warning(warning_msg, "annotation")
                        else:
                            print(f"⚠️ {warning_msg}")
                        continue

                    start_time = time.time()

                    # Procesar imagen
                    results = self._process_single_image(
                        image_path, mask_dir, texts, labels, label_dict
                    )

                    if not results:
                        warning_msg = f"No se procesaron máscaras para {image_file}"
                        if self.logger:
                            self.logger.log_warning(warning_msg, "annotation")
                        continue

                    processing_time = time.time() - start_time

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

                    # Análisis específico de avocados
                    if self.avocado_analytics:
                        analysis = self.avocado_analytics.analyze_avocado_detection(
                            image_path, results['detections'], results.get('annotated_masks')
                        )
                        if analysis:
                            analysis_results.append(analysis)

                            # Agregar entrada al timeline
                            self.avocado_analytics.add_processing_timeline_entry(
                                time.time(),
                                analysis['avocado_count'],
                                image_file
                            )

                    processed_images += 1

                    # Log del procesamiento exitoso
                    if self.logger:
                        # Contar detecciones por clase
                        class_counts = {}
                        for det in results['detections']:
                            label = det['label']
                            class_counts[label] = class_counts.get(label, 0) + 1

                        self.logger.log_image_processing(
                            image_file,
                            "annotation",
                            processing_time,
                            detections_made=len(results['detections'])
                        )

                        self.logger.log_annotation_metrics(
                            image_file,
                            len(results.get('annotated_masks', [])),
                            results['detections'],
                            processing_time
                        )

                except Exception as e:
                    if self.logger:
                        self.logger.log_error(f"Error procesando {image_file}: {e}", "annotation", context=image_file)
                    else:
                        print(f"❌ Error procesando {image_file}: {e}")
                    continue

        # Generar analytics de avocados si aplica
        if self.avocado_analytics and analysis_results:
            self._generate_avocado_analytics(analysis_results)

        # Finalizar fase
        if self.logger:
            self.logger.end_phase("annotation")
            self.logger.main_logger.info(f"📊 Resumen de anotación:")
            self.logger.main_logger.info(f"   • Imágenes procesadas: {processed_images}/{total_images}")

            if processed_images < total_images:
                failed_count = total_images - processed_images
                self.logger.main_logger.warning(f"   • Imágenes fallidas: {failed_count}")
        else:
            print(f"✅ Anotación completada: {processed_images}/{total_images} imágenes")

    def _process_single_image(self, image_path, mask_dir, texts, labels, label_dict):
        """Procesa una imagen individual"""
        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.open(image_path).convert('RGB')
            img_width, img_height = pil_image.size

            # Obtener lista de máscaras
            mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
            mask_files.sort()  # Ordenar para consistencia

            detections = []
            yolo_labels = []
            annotated_masks = []

            for mask_file in mask_files:
                mask_path = os.path.join(mask_dir, mask_file)

                # Cargar máscara
                mask = cv2.imread(mask_path, 0)  # Cargar en escala de grises

                # Clasificar máscara
                predicted_label = self.classify_mask(image_rgb, mask, texts, labels)
                label_id = label_dict[predicted_label]

                # Generar detección y etiqueta YOLO
                detection, yolo_label = self._process_mask_region(
                    mask, predicted_label, label_id, img_width, img_height
                )

                if detection and yolo_label:
                    detections.append(detection)
                    yolo_labels.append(yolo_label)

                    # Agregar máscara anotada para visualización
                    annotated_masks.append({
                        'segmentation': mask,
                        'area': np.sum(mask > 0),
                        'label': predicted_label
                    })

            return {
                'detections': detections,
                'yolo_labels': yolo_labels,
                'annotated_masks': annotated_masks,
                'image_rgb': image_rgb
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error procesando imagen {image_path}: {e}", "annotation", context=image_path)
            else:
                print(f"❌ Error procesando imagen {image_path}: {e}")
            return None

    def _process_mask_region(self, mask, label, label_id, img_width, img_height):
        """Procesa una región de máscara para generar detección y etiqueta YOLO"""
        try:
            # Detectar componentes conectados
            labeled_mask, num_labels = label_region(mask)

            if num_labels == 0:
                return None, None

            # Calcular bounding box de toda la máscara
            coords = np.where(mask > 0)
            if len(coords[0]) == 0:
                return None, None

            ymin, ymax = coords[0].min(), coords[0].max()
            xmin, xmax = coords[1].min(), coords[1].max()

            # Crear detección
            detection = {
                'label': label,
                'xmin': int(xmin),
                'ymin': int(ymin),
                'xmax': int(xmax),
                'ymax': int(ymax)
            }

            # Generar etiqueta YOLO (formato: class_id x1 y1 x2 y2 ... polygon)
            yolo_label = self.label_generator.generate_yolo_polygon_label(
                labeled_mask, label_id, img_width, img_height, num_labels
            )

            return detection, yolo_label

        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error procesando región de máscara: {e}", "annotation")
            else:
                print(f"⚠️ Error procesando región de máscara: {e}")
            return None, None

    def _save_yolo_labels(self, yolo_labels, output_path):
        """Guarda etiquetas en formato YOLO"""
        try:
            with open(output_path, 'w') as f:
                for label in yolo_labels:
                    f.write(label + '\n')
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error guardando etiquetas: {e}", "annotation")
            else:
                print(f"⚠️ Error guardando etiquetas: {e}")

    def _generate_visualizations(self, image_path, results, subfolder, image_file,
                                 output_folder, enable_box_visual, enable_color_visual):
        """Genera visualizaciones opcionales"""
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
            if self.logger:
                self.logger.log_warning(f"Error generando visualizaciones: {e}", "annotation")
            else:
                print(f"⚠️ Error generando visualizaciones: {e}")

    def _generate_avocado_analytics(self, analysis_results):
        """Genera analytics específicos para avocados"""
        try:
            if self.logger:
                self.logger.main_logger.info("🥑 Generando analytics de avocados...")

            # Crear gráficas de distribución de madurez
            chart_path = self.avocado_analytics.create_maturity_distribution_chart()
            if chart_path and self.logger:
                self.logger.main_logger.info(f"📊 Gráfica de madurez: {chart_path}")

            # Crear análisis de tamaños
            size_chart_path = self.avocado_analytics.create_size_analysis_chart()
            if size_chart_path and self.logger:
                self.logger.main_logger.info(f"📊 Análisis de tamaños: {size_chart_path}")

            # Crear dashboard de calidad del cultivo
            dashboard_path = self.avocado_analytics.create_crop_quality_dashboard(analysis_results)
            if dashboard_path and self.logger:
                self.logger.main_logger.info(f"📊 Dashboard de calidad: {dashboard_path}")

            # Exportar reporte completo
            report_path = self.avocado_analytics.export_analytics_report(analysis_results)
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

    # Anotar dataset
    annotator.annotate_dataset(
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


if __name__ == "__main__":
    main().path.isdir(subfolder_path):
    continue

print(f"\n📂 Procesando subcarpeta: {subfolder}")

# Obtener lista de imágenes
image_files = [f for f in os.listdir(subfolder_path)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

total_images += len(image_files)

for image_file in image_files:
    try:
        image_name = Path(image_file).stem
        image_path = os.path.join(subfolder_path, image_file)
        mask_dir = os.path.join(mask_folder, subfolder, image_name)

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
        label_output_path = os.path.join(output_folder, 'labels', subfolder, f'{image_name}.txt')
        os.makedirs(os.path.dirname(label_output_path), exist_ok=True)
        self._save_yolo_labels(results['yolo_labels'], label_output_path)

        # Generar visualizaciones si está habilitado
        if enable_visualizations:
            self._generate_visualizations(
                image_path, results, subfolder, image_file, output_folder,
                enable_box_visual, enable_color_visual
            )

        processed_images += 1
        print(f"    ✅ Procesado: {len(results['detections'])} detecciones")

    except Exception as e:
        print(f"    ❌ Error procesando {image_file}: {e}")
        continue

print(f"\n✅ Anotación completada!")
print(f"📊 Procesadas: {processed_images}/{total_images} imágenes")


def _process_single_image(self, image_path, mask_dir, texts, labels, label_dict):
    """Procesa una imagen individual"""
    try:
        # Cargar imagen
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.open(image_path).convert('RGB')
        img_width, img_height = pil_image.size

        # Obtener lista de máscaras
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        mask_files.sort()  # Ordenar para consistencia

        detections = []
        yolo_labels = []
        annotated_masks = []

        for mask_file in mask_files:
            mask_path = os.path.join(mask_dir, mask_file)

            # Cargar máscara
            mask = cv2.imread(mask_path, 0)  # Cargar en escala de grises

            # Clasificar máscara
            predicted_label = self.classify_mask(image_rgb, mask, texts, labels)
            label_id = label_dict[predicted_label]

            # Generar detección y etiqueta YOLO
            detection, yolo_label = self._process_mask_region(
                mask, predicted_label, label_id, img_width, img_height
            )

            if detection and yolo_label:
                detections.append(detection)
                yolo_labels.append(yolo_label)

                # Agregar máscara anotada para visualización
                annotated_masks.append({
                    'segmentation': mask,
                    'area': np.sum(mask > 0),
                    'label': predicted_label
                })

        return {
            'detections': detections,
            'yolo_labels': yolo_labels,
            'annotated_masks': annotated_masks,
            'image_rgb': image_rgb
        }

    except Exception as e:
        print(f"    ❌ Error procesando imagen {image_path}: {e}")
        return None


def _process_mask_region(self, mask, label, label_id, img_width, img_height):
    """Procesa una región de máscara para generar detección y etiqueta YOLO"""
    try:
        # Detectar componentes conectados
        labeled_mask, num_labels = label_region(mask)

        if num_labels == 0:
            return None, None

        # Calcular bounding box de toda la máscara
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return None, None

        ymin, ymax = coords[0].min(), coords[0].max()
        xmin, xmax = coords[1].min(), coords[1].max()

        # Crear detección
        detection = {
            'label': label,
            'xmin': int(xmin),
            'ymin': int(ymin),
            'xmax': int(xmax),
            'ymax': int(ymax)
        }

        # Generar etiqueta YOLO (formato: class_id x1 y1 x2 y2 ... polygon)
        yolo_label = self.label_generator.generate_yolo_polygon_label(
            labeled_mask, label_id, img_width, img_height, num_labels
        )

        return detection, yolo_label

    except Exception as e:
        print(f"    ⚠️ Error procesando región de máscara: {e}")
        return None, None


def _save_yolo_labels(self, yolo_labels, output_path):
    """Guarda etiquetas en formato YOLO"""
    try:
        with open(output_path, 'w') as f:
            for label in yolo_labels:
                f.write(label + '\n')
    except Exception as e:
        print(f"    ⚠️ Error guardando etiquetas: {e}")


def _generate_visualizations(self, image_path, results, subfolder, image_file,
                             output_folder, enable_box_visual, enable_color_visual):
    """Genera visualizaciones opcionales"""
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

    args = parser.parse_args()

    # Crear anotador
    annotator = CLIPAnnotator()

    # Anotar dataset
    annotator.annotate_dataset(
        image_folder=args.image_folder,
        mask_folder=args.mask_folder,
        description_file=args.description_file,
        output_folder=args.output_folder,
        enable_visualizations=True,
        enable_box_visual=args.box_visual,
        enable_color_visual=args.color_visual
    )


if __name__ == "__main__":
    main()