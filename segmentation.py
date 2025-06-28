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
import torch.multiprocessing as mp
from itertools import repeat

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

        # Inicializar procesador de máscaras (sin pasar logger)
        self.mask_processor = MaskProcessor()

    def _init_model(self, device_id=0):
        """Inicializa el modelo SAM2 en una GPU específica."""
        try:
            # Configurar GPU si está disponible
            if torch.cuda.is_available():
                torch.cuda.set_device(device_id)
                device = f'cuda:{device_id}'

                # Optimizaciones para GPU moderna
                torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
                if torch.cuda.get_device_properties(device_id).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            else:
                device = 'cpu'

            # Construir modelo SAM2
            sam2 = build_sam2(
                self.model_cfg,
                self.checkpoint_path,
                device=device,
                apply_postprocessing=False
            )

            # Crear generador de máscaras automático
            mask_generator = SAM2AutomaticMaskGenerator(
                sam2,
                points_per_side=self.points_per_side,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=self.min_mask_region_area
            )

            if self.logger:
                self.logger.main_logger.info(f"✅ Modelo SAM2 inicializado en {device}")
            else:
                print(f"✅ Modelo SAM2 inicializado en {device}")
            
            return sam2, mask_generator

        except Exception as e:
            error_msg = f"Error inicializando modelo SAM2: {e}"
            if self.logger:
                self.logger.log_error(error_msg, "segmentation")
            else:
                print(f"❌ {error_msg}")
            raise

    def segment_image(self, image_path, mask_generator):
        """
        Segmenta una imagen individual

        Args:
            image_path (str): Ruta de la imagen
            mask_generator: El generador de mascaras de SAM2

        Returns:
            list: Lista de máscaras con metadatos
        """
        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generar máscaras
            masks = mask_generator.generate(image_rgb)

            if self.logger:
                self.logger.main_logger.info(f"🎭 Generadas {len(masks)} máscaras para {Path(image_path).name}")
            else:
                print(f"🎭 Generadas {len(masks)} máscaras para {Path(image_path).name}")

            return masks

        except Exception as e:
            error_msg = f"Error segmentando {image_path}: {e}"
            if self.logger:
                self.logger.log_error(
                    error_msg,
                    "segmentation",
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
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Found {num_gpus} GPUs. Running in parallel mode.")
            self._segment_dataset_parallel(image_folder, output_folder, enable_mask_nms, mask_nms_thresh, save_annotations, save_json, num_gpus)
        else:
            print("Found 1 or 0 GPUs. Running in sequential mode.")
            self._segment_dataset_sequential(image_folder, output_folder, enable_mask_nms, mask_nms_thresh, save_annotations, save_json)

    def _segment_dataset_sequential(self, image_folder, output_folder, enable_mask_nms, mask_nms_thresh, save_annotations, save_json):
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
        sam2, mask_generator = self._init_model()

        # Procesar cada subcarpeta (train, val, test)
        for subfolder_name, subfolder_info in dataset_structure['subfolders'].items():
            if self.logger:
                self.logger.main_logger.info(f"📂 Procesando subcarpeta: {subfolder_name}")

            for image_file in subfolder_info['image_files']:
                image_path = os.path.join(subfolder_info['path'], image_file)
                self._process_single_image(image_path, subfolder_name, output_folder, enable_mask_nms, mask_nms_thresh, save_annotations, save_json, mask_generator)
                processed_images += 1

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

    def _segment_dataset_parallel(self, image_folder, output_folder, enable_mask_nms, mask_nms_thresh, save_annotations, save_json, num_gpus):
        file_manager = FileManager(output_folder)
        file_manager.create_output_structure()
        dataset_structure = file_manager.organize_dataset_structure(image_folder)
        all_tasks = []
        for subfolder_name, subfolder_info in dataset_structure['subfolders'].items():
            for image_file in subfolder_info['image_files']:
                image_path = os.path.join(subfolder_info['path'], image_file)
                all_tasks.append((image_path, subfolder_name, output_folder, enable_mask_nms, mask_nms_thresh, save_annotations, save_json))

        # Assign tasks to GPUs
        tasks_by_gpu = [[] for _ in range(num_gpus)]
        for i, task in enumerate(all_tasks):
            tasks_by_gpu[i % num_gpus].append(task)

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_gpus) as pool:
            pool.starmap(self._process_batch, zip(range(num_gpus), tasks_by_gpu))

    def _process_batch(self, device_id, tasks):
        sam2, mask_generator = self._init_model(device_id)
        for task in tasks:
            self._process_single_image(*task, mask_generator)

    def _process_single_image(self, image_path, subfolder_name, output_folder, enable_mask_nms, mask_nms_thresh, save_annotations, save_json, mask_generator):
        try:
            image_name = Path(image_path).stem
            start_time = time.time()

            # Segmentar imagen
            masks = self.segment_image(image_path, mask_generator)

            if not masks:
                if self.logger:
                    self.logger.log_warning(f"No se generaron máscaras para {image_name}", "segmentation")
                return

            original_mask_count = len(masks)

            # Aplicar Mask NMS si está habilitado
            if enable_mask_nms:
                masks = self.mask_processor.apply_mask_nms(masks, mask_nms_thresh)

                if self.logger:
                    reduction_pct = ((original_mask_count - len(
                        masks)) / original_mask_count) * 100 if original_mask_count > 0 else 0
                    self.logger.seg_logger.info(
                        f"🎯 NMS aplicado a {image_name}: {original_mask_count} → {len(masks)} máscaras (-{reduction_pct:.1f}%)")

            # Guardar máscaras
            mask_output_dir = os.path.join(output_folder, 'mask', subfolder_name, image_name)
            os.makedirs(mask_output_dir, exist_ok=True)

            self._save_masks(masks, mask_output_dir)

            # Guardar visualización con índices
            if save_annotations:
                visual_output_path = os.path.join(output_folder, 'mask_idx_visual', subfolder_name, f"{image_name}.png")
                os.makedirs(os.path.dirname(visual_output_path), exist_ok=True)
                self._save_mask_visualization(masks, image_path, visual_output_path)

            # Guardar metadatos JSON
            if save_json:
                json_output_dir = os.path.join(output_folder, 'json', subfolder_name, image_name)
                os.makedirs(json_output_dir, exist_ok=True)
                self._save_mask_metadata(masks, json_output_dir)

            processing_time = time.time() - start_time

            # Log del procesamiento exitoso
            if self.logger:
                self.logger.log_image_processing(
                    image_name,
                    "segmentation",
                    processing_time,
                    masks_generated=len(masks)
                )

                # Actualizar métricas de segmentación
                self.logger.log_segmentation_metrics(
                    image_name,
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
                self.logger.log_error(f"Error procesando {image_name}: {e}", "segmentation", context=image_name)
            else:
                print(f"❌ Error procesando {image_name}: {e}")

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

    parser = argparse.ArgumentParser(description='Módulo de segmentación SAM2')
    parser.add_argument('--image_path', type=str, required=True, help='Ruta de la imagen a segmentar')
    parser.add_argument('--output_folder', type=str, default='./test_output', help='Carpeta de salida')
    parser.add_argument('--enable_nms', action='store_true', help='Aplicar NMS a las máscaras')

    args = parser.parse_args()

    # Crear segmentador
    segmentator = SAM2Segmentator()

    # Segmentar imagen
    masks = segmentator.segment_image(args.image_path)

    if args.enable_nms:
        masks = segmentator.mask_processor.apply_mask_nms(masks)

    print(f"✅ Segmentación completada: {len(masks)} máscaras")


if __name__ == "__main__":
    main()