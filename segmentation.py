"""
Módulo de Segmentación - SDM-D Framework
Maneja la segmentación de imágenes usando SAM2 y el procesamiento de máscaras.
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import time

# Importar SAM2
sys.path.insert(0, os.path.join(os.getcwd(), 'sam2'))
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from utiles.mask_utils import MaskProcessor
from utiles.file_utils import FileManager
from utiles.logging_utils import SDMLogger


class SAM2Segmentator:
    """
    Clase principal para manejar la segmentación con SAM2
    """

    def __init__(self, model_cfg="sam2_hiera_l.yaml", checkpoint_path="./checkpoints/sam2_hiera_large.pt",
                 points_per_side=32, min_mask_region_area=50, logger=None):
        """
        Inicializa el segmentador SAM2

        Args:
            model_cfg (str): Archivo de configuración del modelo
            checkpoint_path (str): Ruta al checkpoint del modelo
            points_per_side (int): Número de puntos por lado para la grilla
            min_mask_region_area (int): Área mínima de región para las máscaras
            logger (SDMLogger): Logger para seguimiento
        """
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        self.points_per_side = points_per_side
        self.min_mask_region_area = min_mask_region_area

        # Logger
        self.logger = logger

        # Inicializar modelo
        self._init_model()

        # Inicializar procesador de máscaras
        self.mask_processor = MaskProcessor()

    def _init_model(self):
        """Inicializa el modelo SAM2"""
        try:
            # Configurar GPU si está disponible
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                device = 'cuda'

                # Optimizaciones para GPU moderna
                torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            else:
                device = 'cpu'

            # Construir modelo SAM2
            self.sam2 = build_sam2(
                self.model_cfg,
                self.checkpoint_path,
                device=device,
                apply_postprocessing=False
            )

            # Crear generador de máscaras automático
            self.mask_generator = SAM2AutomaticMaskGenerator(
                self.sam2,
                points_per_side=self.points_per_side,
                min_mask_region_area=self.min_mask_region_area
            )

            if self.logger:
                self.logger.main_logger.info(f"✅ SAM2 inicializado correctamente en {device}")
            else:
                print(f"✅ SAM2 inicializado correctamente en {device}")

        except Exception as e:
            error_msg = f"Error inicializando SAM2: {e}"
            if self.logger:
                self.logger.log_error(error_msg, "segmentation")
            else:
                print(f"❌ {error_msg}")
            raise

    def segment_image(self, image_path):
        """
        Segmenta una imagen individual

        Args:
            image_path (str): Ruta a la imagen

        Returns:
            list: Lista de máscaras generadas
        """
        start_time = time.time()

        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generar máscaras
            masks = self.mask_generator.generate(image_rgb)

            # Ordenar por área (descendente)
            sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

            processing_time = time.time() - start_time

            # Log del procesamiento
            if self.logger:
                self.logger.log_segmentation_metrics(
                    os.path.basename(image_path),
                    len(sorted_masks),
                    len(sorted_masks),  # Será actualizado si se aplica NMS
                    False,  # NMS no aplicado aún
                    processing_time
                )

            return sorted_masks

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error segmentando imagen {image_path}: {e}"

            if self.logger:
                self.logger.log_error(error_msg, "segmentation", context=image_path)
                self.logger.log_image_processing(
                    os.path.basename(image_path),
                    "segmentation",
                    processing_time,
                    error=e
                )
            else:
                print(f"❌ {error_msg}")

            return []

    def segment_dataset(self, image_folder, output_folder, enable_mask_nms=True,
                        mask_nms_thresh=0.9, save_annotations=True, save_json=False):
        """
        Segmenta un dataset completo de imágenes

        Args:
            image_folder (str): Carpeta con las imágenes
            output_folder (str): Carpeta de salida
            enable_mask_nms (bool): Aplicar NMS a las máscaras
            mask_nms_thresh (float): Umbral para NMS
            save_annotations (bool): Guardar visualizaciones
            save_json (bool): Guardar metadatos JSON
        """

        # Crear estructura de carpetas
        file_manager = FileManager(output_folder)
        file_manager.create_output_structure()

        # Analizar dataset
        dataset_structure = file_manager.organize_dataset_structure(image_folder)
        total_images = dataset_structure['total_images']

        if self.logger:
            self.logger.start_phase("segmentation", total_images)
            self.logger.main_logger.info(f"📁 Carpetas a procesar: {list(dataset_structure['subfolders'].keys())}")

        processed_images = 0

        # Procesar cada subcarpeta (train, val, test)
        for subfolder_name, subfolder_info in dataset_structure['subfolders'].items():
            if self.logger:
                self.logger.main_logger.info(f"📂 Procesando subcarpeta: {subfolder_name}")

            for image_file in subfolder_info['image_files']:
                try:
                    image_path = os.path.join(subfolder_info['path'], image_file)
                    image_name = Path(image_file).stem

                    start_time = time.time()

                    # Segmentar imagen
                    masks = self.segment_image(image_path)

                    if not masks:
                        if self.logger:
                            self.logger.log_warning(f"No se generaron máscaras para {image_file}", "segmentation")
                        continue

                    original_mask_count = len(masks)

                    # Aplicar Mask NMS si está habilitado
                    if enable_mask_nms:
                        masks = self.mask_processor.apply_mask_nms(masks, mask_nms_thresh)

                        if self.logger:
                            reduction_pct = ((original_mask_count - len(
                                masks)) / original_mask_count) * 100 if original_mask_count > 0 else 0
                            self.logger.seg_logger.info(
                                f"🎯 NMS aplicado a {image_file}: {original_mask_count} → {len(masks)} máscaras (-{reduction_pct:.1f}%)")

                    # Guardar máscaras
                    mask_output_dir = os.path.join(output_folder, 'mask', subfolder_name, image_name)
                    os.makedirs(mask_output_dir, exist_ok=True)

                    self._save_masks(masks, mask_output_dir)

                    # Guardar visualización con índices
                    if save_annotations:
                        visual_output_path = os.path.join(output_folder, 'mask_idx_visual', subfolder_name, image_file)
                        os.makedirs(os.path.dirname(visual_output_path), exist_ok=True)
                        self._save_mask_visualization(masks, image_path, visual_output_path)

                    # Guardar metadatos JSON
                    if save_json:
                        json_output_dir = os.path.join(output_folder, 'json', subfolder_name, image_name)
                        os.makedirs(json_output_dir, exist_ok=True)
                        self._save_mask_metadata(masks, json_output_dir)

                    processing_time = time.time() - start_time
                    processed_images += 1

                    # Log del procesamiento exitoso
                    if self.logger:
                        self.logger.log_image_processing(
                            image_file,
                            "segmentation",
                            processing_time,
                            masks_generated=len(masks)
                        )

                        # Actualizar métricas de segmentación
                        self.logger.log_segmentation_metrics(
                            image_file,
                            original_mask_count,
                            len(masks),
                            enable_mask_nms,
                            processing_time
                        )

                    # Limpiar memoria
                    del masks
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                except Exception as e:
                    if self.logger:
                        self.logger.log_error(f"Error procesando {image_file}: {e}", "segmentation", context=image_file)
                    else:
                        print(f"❌ Error procesando {image_file}: {e}")
                    continue

        # Finalizar fase
        if self.logger:
            self.logger.end_phase("segmentation")
            self.logger.main_logger.info(f"📊 Resumen de segmentación:")
            self.logger.main_logger.info(f"   • Imágenes procesadas: {processed_images}/{total_images}")

            if processed_images < total_images:
                failed_count = total_images - processed_images
                self.logger.main_logger.warning(f"   • Imágenes fallidas: {failed_count}")
        else:
            print(f"✅ Segmentación completada: {processed_images}/{total_images} imágenes")

    def _save_masks(self, masks, output_dir):
        """Guarda las máscaras como imágenes PNG"""
        for i, mask in enumerate(masks):
            mask_img = mask['segmentation']
            mask_img = np.stack([mask_img] * 3, axis=-1)  # Convertir a 3 canales
            mask_img = (mask_img * 255).astype(np.uint8)  # Convertir a blanco

            output_path = os.path.join(output_dir, f'mask_{i}.png')
            cv2.imwrite(output_path, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))

    def _save_mask_visualization(self, masks, image_path, output_path):
        """Guarda visualización de máscaras con índices"""
        try:
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Usar utilidad de visualización
            visual_image = self.mask_processor.create_indexed_visualization(masks, image_rgb)

            # Guardar
            cv2.imwrite(output_path, cv2.cvtColor(visual_image, cv2.COLOR_RGB2BGR))

        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error guardando visualización: {e}", "segmentation")
            else:
                print(f"⚠️ Error guardando visualización: {e}")

    def _save_mask_metadata(self, masks, output_dir):
        """Guarda metadatos de máscaras en formato JSON"""
        import json

        for i, mask in enumerate(masks):
            # Extraer metadatos relevantes (sin la máscara binaria)
            metadata = {
                "area": mask['area'],
                "bbox": mask['bbox'],
                "predicted_iou": mask['predicted_iou'],
                "point_coords": mask['point_coords'].tolist() if hasattr(mask['point_coords'], 'tolist') else mask[
                    'point_coords'],
                "stability_score": mask['stability_score'],
                "crop_box": mask['crop_box']
            }

            output_path = os.path.join(output_dir, f'mask_{i}.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    """Función principal para testing del módulo"""
    import argparse

    parser = argparse.ArgumentParser(description='Módulo de Segmentación SAM2')
    parser.add_argument('--image_folder', type=str, required=True, help='Carpeta con imágenes')
    parser.add_argument('--output_folder', type=str, required=True, help='Carpeta de salida')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/sam2_hiera_large.pt', help='Checkpoint SAM2')
    parser.add_argument('--enable_nms', action='store_true', help='Aplicar Mask NMS')
    parser.add_argument('--nms_thresh', type=float, default=0.9, help='Umbral NMS')
    parser.add_argument('--save_json', action='store_true', help='Guardar metadatos JSON')
    parser.add_argument('--verbose', action='store_true', help='Logging detallado')

    args = parser.parse_args()

    # Configurar logger si está en modo verbose
    logger = None
    if args.verbose:
        logger = SDMLogger(args.output_folder, enable_console=True)

    # Crear segmentador
    segmentator = SAM2Segmentator(checkpoint_path=args.checkpoint, logger=logger)

    # Procesar dataset
    segmentator.segment_dataset(
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        enable_mask_nms=args.enable_nms,
        mask_nms_thresh=args.nms_thresh,
        save_json=args.save_json
    )

    # Guardar reporte si hay logger
    if logger:
        logger.save_session_report()


if __name__ == "__main__":
    main()
    ación
    completada!")
    print(f"📊 Procesadas: {processed_images}/{total_images} imágenes")


def _save_masks(self, masks, output_dir):
    """Guarda las máscaras como imágenes PNG"""
    for i, mask in enumerate(masks):
        mask_img = mask['segmentation']
        mask_img = np.stack([mask_img] * 3, axis=-1)  # Convertir a 3 canales
        mask_img = (mask_img * 255).astype(np.uint8)  # Convertir a blanco

        output_path = os.path.join(output_dir, f'mask_{i}.png')
        cv2.imwrite(output_path, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))


def _save_mask_visualization(self, masks, image_path, output_path):
    """Guarda visualización de máscaras con índices"""
    try:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Usar utilidad de visualización
        visual_image = self.mask_processor.create_indexed_visualization(masks, image_rgb)

        # Guardar
        cv2.imwrite(output_path, cv2.cvtColor(visual_image, cv2.COLOR_RGB2BGR))

    except Exception as e:
        print(f"    ⚠️ Error guardando visualización: {e}")


def _save_mask_metadata(self, masks, output_dir):
    """Guarda metadatos de máscaras en formato JSON"""
    import json

    for i, mask in enumerate(masks):
        # Extraer metadatos relevantes (sin la máscara binaria)
        metadata = {
            "area": mask['area'],
            "bbox": mask['bbox'],
            "predicted_iou": mask['predicted_iou'],
            "point_coords": mask['point_coords'].tolist() if hasattr(mask['point_coords'], 'tolist') else mask[
                'point_coords'],
            "stability_score": mask['stability_score'],
            "crop_box": mask['crop_box']
        }

        output_path = os.path.join(output_dir, f'mask_{i}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    """Función principal para testing del módulo"""
    import argparse

    parser = argparse.ArgumentParser(description='Módulo de Segmentación SAM2')
    parser.add_argument('--image_folder', type=str, required=True, help='Carpeta con imágenes')
    parser.add_argument('--output_folder', type=str, required=True, help='Carpeta de salida')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/sam2_hiera_large.pt', help='Checkpoint SAM2')
    parser.add_argument('--enable_nms', action='store_true', help='Aplicar Mask NMS')
    parser.add_argument('--nms_thresh', type=float, default=0.9, help='Umbral NMS')
    parser.add_argument('--save_json', action='store_true', help='Guardar metadatos JSON')

    args = parser.parse_args()

    # Crear segmentador
    segmentator = SAM2Segmentator(checkpoint_path=args.checkpoint)

    # Procesar dataset
    segmentator.segment_dataset(
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        enable_mask_nms=args.enable_nms,
        mask_nms_thresh=args.nms_thresh,
        save_json=args.save_json
    )


if __name__ == "__main__":
    main()