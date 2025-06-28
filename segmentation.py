#!/usr/bin/env python3
"""
Módulo de Segmentación usando SAM2
Compatible con Python 3.12
Integración con sistema de logging y manejo optimizado de memoria
"""

import os
import cv2
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Importaciones SAM2
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:
    raise ImportError(f"Error importando SAM2: {e}. Verifica la instalación de SAM2.")

# Importaciones locales
from utiles.mask_utils import MaskProcessor
from utiles.file_manager import FileManager
from utiles.logging_utils import SDMLogger


class SAM2Segmentator:
    """
    Clase para segmentación usando SAM2 con logging y optimizaciones
    """

    def __init__(self, checkpoint_path: str = "./checkpoints/sam2_hiera_large.pt",
                 config_name: str = "sam2_hiera_l.yaml",
                 points_per_side: int = 32,
                 pred_iou_thresh: float = 0.8,
                 stability_score_thresh: float = 0.95,
                 min_mask_region_area: int = 100,
                 logger: Optional[SDMLogger] = None):
        """
        Inicializa el segmentador SAM2

        Args:
            checkpoint_path (str): Ruta al checkpoint de SAM2
            config_name (str): Nombre del archivo de configuración
            points_per_side (int): Puntos por lado para la grid
            pred_iou_thresh (float): Umbral de IoU predicho
            stability_score_thresh (float): Umbral de score de estabilidad
            min_mask_region_area (int): Área mínima de región de máscara
            logger (Optional[SDMLogger]): Logger personalizado
        """
        self.checkpoint_path = checkpoint_path
        self.config_name = config_name
        self.logger = logger

        # Parámetros de segmentación
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area

        # Inicializar procesador de máscaras
        self.mask_processor = MaskProcessor(logger=logger)

        # Configurar dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.logger:
            self.logger.main_logger.info(f"🔧 Dispositivo seleccionado: {self.device}")
            self.logger.main_logger.info(f"🔧 Configuración SAM2: {config_name}")
        else:
            print(f"🔧 Dispositivo seleccionado: {self.device}")

        # Inicializar SAM2
        self._initialize_sam2()

    def _initialize_sam2(self):
        """Inicializa el modelo SAM2"""
        try:
            if self.logger:
                self.logger.main_logger.info("🔄 Inicializando SAM2...")
            else:
                print("🔄 Inicializando SAM2...")

            # Verificar checkpoint
            if not os.path.exists(self.checkpoint_path):
                error_msg = f"Checkpoint no encontrado: {self.checkpoint_path}"
                if self.logger:
                    self.logger.log_error(error_msg, "segmentation")
                else:
                    print(f"❌ {error_msg}")
                raise FileNotFoundError(error_msg)

            # Construir modelo SAM2
            self.sam2_model = build_sam2(
                config_file=self.config_name,
                ckpt_path=self.checkpoint_path,
                device=self.device
            )

            # Crear generador automático de máscaras
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2_model,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                min_mask_region_area=self.min_mask_region_area,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
            )

            if self.logger:
                self.logger.main_logger.info("✅ SAM2 inicializado correctamente")
                self.logger.main_logger.info(f"📊 Parámetros: points_per_side={self.points_per_side}, "
                                             f"iou_thresh={self.pred_iou_thresh}, "
                                             f"stability_thresh={self.stability_score_thresh}")
            else:
                print("✅ SAM2 inicializado correctamente")

        except Exception as e:
            error_msg = f"Error inicializando SAM2: {e}"
            if self.logger:
                self.logger.log_error(error_msg, "segmentation")
            else:
                print(f"❌ {error_msg}")
            raise

    def segment_image(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Segmenta una imagen individual

        Args:
            image_path (str): Ruta a la imagen

        Returns:
            List[Dict]: Lista de máscaras con metadatos
        """
        try:
            start_time = time.time()

            # Cargar imagen
            if not os.path.exists(image_path):
                error_msg = f"Imagen no encontrada: {image_path}"
                if self.logger:
                    self.logger.log_error(error_msg, "segmentation")
                else:
                    print(f"❌ {error_msg}")
                return []

            image = cv2.imread(image_path)
            if image is None:
                error_msg = f"No se pudo cargar la imagen: {image_path}"
                if self.logger:
                    self.logger.log_error(error_msg, "segmentation")
                else:
                    print(f"❌ {error_msg}")
                return []

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.logger:
                self.logger.main_logger.debug(f"🖼️ Procesando imagen: {os.path.basename(image_path)} "
                                              f"({image_rgb.shape[1]}x{image_rgb.shape[0]})")

            # Generar máscaras
            masks = self.mask_generator.generate(image_rgb)

            processing_time = time.time() - start_time

            if self.logger:
                self.logger.log_processing_time(
                    "segmentation",
                    processing_time,
                    metadata={
                        "image_path": image_path,
                        "image_size": f"{image_rgb.shape[1]}x{image_rgb.shape[0]}",
                        "masks_generated": len(masks)
                    }
                )
                self.logger.main_logger.debug(f"✅ Generadas {len(masks)} máscaras en {processing_time:.2f}s")
            else:
                print(f"✅ Generadas {len(masks)} máscaras en {processing_time:.2f}s")

            return masks

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error segmentando imagen {image_path}: {e}"

            if self.logger:
                self.logger.log_processing_time(
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
                        if self.logger:
                            self.logger.main_logger.debug(f"🔄 Aplicando NMS con umbral {mask_nms_thresh}")

                        masks = self.mask_processor.apply_mask_nms(masks, mask_nms_thresh)

                        if self.logger:
                            removed_count = original_mask_count - len(masks)
                            self.logger.main_logger.debug(f"📊 NMS: {original_mask_count} → {len(masks)} "
                                                          f"(removidas: {removed_count})")

                    # Crear directorios de salida específicos
                    output_structure = file_manager.get_output_structure_for_dataset()

                    image_output_dir = os.path.join(
                        output_structure['masks']['path'],
                        subfolder_name
                    )
                    os.makedirs(image_output_dir, exist_ok=True)

                    # Guardar máscaras como imágenes PNG
                    self._save_masks(masks, image_output_dir, image_name)

                    # Guardar visualización si está habilitado
                    if save_annotations:
                        visualization_dir = os.path.join(
                            output_structure['visualizations']['path'],
                            subfolder_name
                        )
                        os.makedirs(visualization_dir, exist_ok=True)

                        visualization_path = os.path.join(visualization_dir, f"{image_name}_segmentation.png")
                        self._save_mask_visualization(masks, image_path, visualization_path)

                    # Guardar metadatos JSON si está habilitado
                    if save_json:
                        metadata_dir = os.path.join(
                            output_structure['metadata']['path'],
                            subfolder_name
                        )
                        os.makedirs(metadata_dir, exist_ok=True)

                        metadata_file = os.path.join(metadata_dir, f"{image_name}_metadata.json")
                        self._save_mask_metadata(masks, metadata_file)

                    processing_time = time.time() - start_time
                    processed_images += 1

                    if self.logger:
                        self.logger.main_logger.info(f"✅ {image_file}: {len(masks)} máscaras "
                                                     f"({processing_time:.2f}s)")
                        self.logger.update_progress("segmentation", processed_images)

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

    def _save_masks(self, masks, output_dir, image_name):
        """Guarda las máscaras como imágenes PNG"""
        for i, mask in enumerate(masks):
            mask_img = mask['segmentation']
            mask_img = np.stack([mask_img] * 3, axis=-1)  # Convertir a 3 canales
            mask_img = (mask_img * 255).astype(np.uint8)  # Convertir a blanco

            output_path = os.path.join(output_dir, f'{image_name}_mask_{i}.png')
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

    def _save_mask_metadata(self, masks, output_file):
        """Guarda metadatos de máscaras en formato JSON"""
        import json

        metadata_list = []
        for i, mask in enumerate(masks):
            # Extraer metadatos relevantes (sin la máscara binaria)
            metadata = {
                "mask_id": i,
                "area": int(mask['area']),
                "bbox": mask['bbox'],
                "predicted_iou": float(mask['predicted_iou']),
                "point_coords": mask['point_coords'].tolist() if hasattr(mask['point_coords'], 'tolist') else mask[
                    'point_coords'],
                "stability_score": float(mask['stability_score']),
                "crop_box": mask['crop_box']
            }
            metadata_list.append(metadata)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)


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