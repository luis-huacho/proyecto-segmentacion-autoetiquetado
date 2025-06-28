"""
Utilidades para procesamiento de máscaras
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


class MaskProcessor:
    """Clase para procesar y manipular máscaras"""

    def __init__(self):
        pass

    def apply_mask_nms(self, masks, threshold=0.9):
        """
        Aplica Non-Maximum Suppression a las máscaras

        Args:
            masks (list): Lista de máscaras con metadatos
            threshold (float): Umbral de solapamiento

        Returns:
            list: Máscaras filtradas
        """
        if not masks:
            return masks

        # Verificar compatibilidad con numpy
        if np.__version__ >= '1.20':
            bool_type = np.bool_
        else:
            bool_type = np.bool

        # Convertir máscaras a arrays numpy booleanos
        masks_np = [np.array(mask['segmentation'], dtype=bool_type) for mask in masks]
        areas = [np.sum(mask_np) for mask_np in masks_np]
        scores = [mask['stability_score'] for mask in masks]
        keep = torch.ones(len(masks_np), dtype=torch.bool)

        # Aplicar NMS
        for i in range(len(masks_np)):
            if not keep[i]:
                continue

            for j in range(i + 1, len(masks_np)):
                if not keep[j]:
                    continue

                # Calcular intersección
                intersection = np.logical_and(masks_np[i], masks_np[j]).astype(np.float32).sum()
                smaller_area = min(areas[i], areas[j])

                # Si hay solapamiento significativo, mantener la máscara con mejor score
                if intersection > threshold * smaller_area:
                    if scores[i] < scores[j]:
                        keep[i] = False
                    else:
                        keep[j] = False

        # Filtrar máscaras
        filtered_masks = [mask for idx, mask in enumerate(masks) if keep[idx]]
        return filtered_masks

    def calculate_mask_iou(self, mask1, mask2):
        """
        Calcula IoU entre dos máscaras

        Args:
            mask1 (np.ndarray): Primera máscara
            mask2 (np.ndarray): Segunda máscara

        Returns:
            float: Valor IoU
        """
        intersection = np.logical_and(mask1, mask2).astype(np.float32).sum()
        union = np.logical_or(mask1, mask2).astype(np.float32).sum()
        return intersection / union if union > 0 else 0.0

    def create_indexed_visualization(self, masks, image, borders=True):
        """
        Crea visualización de máscaras con índices

        Args:
            masks (list): Lista de máscaras
            image (np.ndarray): Imagen original
            borders (bool): Dibujar bordes

        Returns:
            np.ndarray: Imagen con visualización
        """
        if len(masks) == 0:
            return image

        # Crear imagen de visualización
        img_vis = image.copy().astype(np.float32) / 255.0

        for i, mask in enumerate(masks):
            m = mask['segmentation']

            # Generar color aleatorio para la máscara
            color_mask = np.concatenate([np.random.random(3), [0.5]])

            # Aplicar color a la región de la máscara
            img_vis[m] = img_vis[m] * (1 - color_mask[3]) + color_mask[:3] * color_mask[3]

            # Dibujar bordes si está habilitado
            if borders:
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img_vis, contours, -1, (0, 0, 1), thickness=2)

            # Agregar índice en el centro de la máscara
            moments = cv2.moments(m.astype(np.uint8))
            if moments["m00"] != 0:
                x = int(moments["m10"] / moments["m00"])
                y = int(moments["m01"] / moments["m00"])
                cv2.putText(img_vis, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (1, 1, 1), 2, cv2.LINE_AA)

        # Asegurar que los valores estén en rango [0, 1]
        img_vis = np.clip(img_vis, 0, 1)

        # Convertir de vuelta a uint8
        return (img_vis * 255).astype(np.uint8)

    def save_masks_as_images(self, masks, output_dir):
        """
        Guarda máscaras como imágenes individuales

        Args:
            masks (list): Lista de máscaras
            output_dir (str): Directorio de salida
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        for i, mask in enumerate(masks):
            mask_img = mask['segmentation']
            mask_img = np.stack([mask_img] * 3, axis=-1)  # Convertir a 3 canales
            mask_img = (mask_img * 255).astype(np.uint8)

            output_path = os.path.join(output_dir, f'mask_{i}.png')
            cv2.imwrite(output_path, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))

    def extract_mask_contours(self, mask, img_width, img_height, max_points=300):
        """
        Extrae contornos de una máscara y los convierte a coordenadas normalizadas

        Args:
            mask (np.ndarray): Máscara binaria
            img_width (int): Ancho de la imagen
            img_height (int): Alto de la imagen
            max_points (int): Número máximo de puntos del contorno

        Returns:
            list: Lista de coordenadas normalizadas
        """
        # Encontrar contornos
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return []

        # Tomar el contorno más grande
        contour = max(contours, key=cv2.contourArea)
        contour = contour.reshape(-1, 2)

        # Reducir número de puntos si es necesario
        num_points = len(contour)
        if num_points > max_points:
            skip = num_points // max_points
            skip = max(1, skip)
            contour = contour[::skip]

        # Ordenar puntos empezando por el más bajo
        bottom_point_index = np.argmax(contour[:, 1])
        sorted_points = np.concatenate([contour[bottom_point_index:], contour[:bottom_point_index]])

        # Normalizar coordenadas
        normalized_points = []
        for point in sorted_points:
            x_norm = point[0] / img_width
            y_norm = point[1] / img_height
            normalized_points.extend([x_norm, y_norm])

        return normalized_points

    def merge_overlapping_masks(self, masks, iou_threshold=0.5):
        """
        Fusiona máscaras que se solapan significativamente

        Args:
            masks (list): Lista de máscaras
            iou_threshold (float): Umbral IoU para fusión

        Returns:
            list: Lista de máscaras fusionadas
        """
        if len(masks) <= 1:
            return masks

        merged_masks = []
        used_indices = set()

        for i, mask1 in enumerate(masks):
            if i in used_indices:
                continue

            # Buscar máscaras para fusionar
            to_merge = [i]

            for j, mask2 in enumerate(masks[i + 1:], i + 1):
                if j in used_indices:
                    continue

                iou = self.calculate_mask_iou(mask1['segmentation'], mask2['segmentation'])
                if iou > iou_threshold:
                    to_merge.append(j)

            # Fusionar máscaras
            if len(to_merge) == 1:
                merged_masks.append(mask1)
            else:
                merged_mask = self._merge_mask_group([masks[idx] for idx in to_merge])
                merged_masks.append(merged_mask)

            # Marcar como usadas
            used_indices.update(to_merge)

        return merged_masks

    def _merge_mask_group(self, mask_group):
        """
        Fusiona un grupo de máscaras en una sola

        Args:
            mask_group (list): Grupo de máscaras a fusionar

        Returns:
            dict: Máscara fusionada
        """
        # Crear máscara combinada
        combined_mask = np.zeros_like(mask_group[0]['segmentation'], dtype=bool)
        total_area = 0
        total_score = 0

        for mask in mask_group:
            combined_mask = np.logical_or(combined_mask, mask['segmentation'])
            total_area += mask['area']
            total_score += mask.get('stability_score', 0)

        # Calcular nuevo bounding box
        coords = np.where(combined_mask)
        if len(coords[0]) > 0:
            ymin, ymax = coords[0].min(), coords[0].max()
            xmin, xmax = coords[1].min(), coords[1].max()
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        else:
            bbox = [0, 0, 0, 0]

        # Crear máscara fusionada
        merged_mask = {
            'segmentation': combined_mask,
            'area': np.sum(combined_mask),
            'bbox': bbox,
            'stability_score': total_score / len(mask_group),
            'predicted_iou': sum(mask.get('predicted_iou', 0) for mask in mask_group) / len(mask_group),
            'point_coords': mask_group[0].get('point_coords', [[0, 0]]),  # Usar primer punto
            'crop_box': bbox
        }

        return merged_mask

    def filter_small_masks(self, masks, min_area=50):
        """
        Filtra máscaras pequeñas

        Args:
            masks (list): Lista de máscaras
            min_area (int): Área mínima

        Returns:
            list: Máscaras filtradas
        """
        return [mask for mask in masks if mask['area'] >= min_area]

    def get_mask_statistics(self, masks):
        """
        Obtiene estadísticas de un conjunto de máscaras

        Args:
            masks (list): Lista de máscaras

        Returns:
            dict: Estadísticas
        """
        if not masks:
            return {
                'count': 0,
                'total_area': 0,
                'avg_area': 0,
                'min_area': 0,
                'max_area': 0,
                'avg_score': 0
            }

        areas = [mask['area'] for mask in masks]
        scores = [mask.get('stability_score', 0) for mask in masks]

        return {
            'count': len(masks),
            'total_area': sum(areas),
            'avg_area': np.mean(areas),
            'min_area': min(areas),
            'max_area': max(areas),
            'avg_score': np.mean(scores) if scores else 0
        }