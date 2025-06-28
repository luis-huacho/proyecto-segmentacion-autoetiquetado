"""
Utilidades para generación de etiquetas y conversiones de formato
"""

import cv2
import numpy as np
from scipy.ndimage import label as label_region


class LabelGenerator:
    """Clase para generar etiquetas en diferentes formatos"""

    def __init__(self):
        pass

    def generate_yolo_polygon_label(self, labeled_mask, class_id, img_width, img_height,
                                    num_labels, max_points=300):
        """
        Genera etiqueta YOLO en formato polígono para segmentación de instancias

        Args:
            labeled_mask (np.ndarray): Máscara con regiones etiquetadas
            class_id (int): ID de la clase
            img_width (int): Ancho de la imagen
            img_height (int): Alto de la imagen
            num_labels (int): Número de regiones conectadas
            max_points (int): Número máximo de puntos del polígono

        Returns:
            str: Etiqueta en formato YOLO
        """
        label_line = f'{class_id}'

        # Procesar cada región conectada
        for region_label in range(1, num_labels + 1):
            # Crear máscara para la región actual
            region_mask = (labeled_mask == region_label).astype(np.uint8) * 255

            # Extraer contorno
            contours, _ = cv2.findContours(region_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            if not contours:
                continue

            # Tomar el contorno más grande
            contour = max(contours, key=cv2.contourArea)
            contour = contour.reshape(-1, 2)

            # Reducir número de puntos si es necesario
            num_points = len(contour)
            if num_points > max_points:
                skip = max(1, num_points // max_points)
                contour = contour[::skip]

            # Ordenar puntos empezando desde el punto más bajo
            bottom_point_index = np.argmax(contour[:, 1])
            sorted_points = np.concatenate([contour[bottom_point_index:], contour[:bottom_point_index]])

            # Normalizar coordenadas y agregar a la etiqueta
            normalized_coords = []
            for point in sorted_points:
                x_norm = point[0] / img_width
                y_norm = point[1] / img_height
                normalized_coords.append(f'{x_norm:.6f}')
                normalized_coords.append(f'{y_norm:.6f}')

            label_line += ' ' + ' '.join(normalized_coords)

        return label_line

    def generate_yolo_bbox_label(self, mask, class_id, img_width, img_height):
        """
        Genera etiqueta YOLO en formato bounding box

        Args:
            mask (np.ndarray): Máscara binaria
            class_id (int): ID de la clase
            img_width (int): Ancho de la imagen
            img_height (int): Alto de la imagen

        Returns:
            str: Etiqueta en formato YOLO bbox
        """
        # Encontrar coordenadas de píxeles activos
        coords = np.where(mask > 0)

        if len(coords[0]) == 0:
            return None

        # Calcular bounding box
        ymin, ymax = coords[0].min(), coords[0].max()
        xmin, xmax = coords[1].min(), coords[1].max()

        # Calcular centro y dimensiones normalizadas
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        return f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}'

    def generate_coco_annotation(self, mask, class_id, annotation_id, image_id, category_info):
        """
        Genera anotación en formato COCO

        Args:
            mask (np.ndarray): Máscara binaria
            class_id (int): ID de la clase
            annotation_id (int): ID de la anotación
            image_id (int): ID de la imagen
            category_info (dict): Información de la categoría

        Returns:
            dict: Anotación COCO
        """
        # Calcular área
        area = int(np.sum(mask > 0))

        if area == 0:
            return None

        # Calcular bounding box
        coords = np.where(mask > 0)
        ymin, ymax = coords[0].min(), coords[0].max()
        xmin, xmax = coords[1].min(), coords[1].max()
        bbox = [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]

        # Extraer segmentación como polígono
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        segmentation = []
        for contour in contours:
            if len(contour) >= 3:  # Mínimo 3 puntos para un polígono
                contour = contour.flatten().tolist()
                segmentation.append(contour)

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": class_id,
            "bbox": bbox,
            "area": area,
            "segmentation": segmentation,
            "iscrowd": 0
        }

        return annotation

    def convert_polygon_to_bbox(self, polygon_coords, img_width, img_height):
        """
        Convierte coordenadas de polígono a bounding box

        Args:
            polygon_coords (list): Lista de coordenadas normalizadas [x1,y1,x2,y2,...]
            img_width (int): Ancho de la imagen
            img_height (int): Alto de la imagen

        Returns:
            tuple: (x_center, y_center, width, height) normalizadas
        """
        if len(polygon_coords) < 4:
            return None

        # Convertir a coordenadas absolutas
        x_coords = [polygon_coords[i] * img_width for i in range(0, len(polygon_coords), 2)]
        y_coords = [polygon_coords[i] * img_height for i in range(1, len(polygon_coords), 2)]

        # Calcular bounding box
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)

        # Normalizar centro y dimensiones
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        return x_center, y_center, width, height

    def convert_bbox_to_polygon(self, x_center, y_center, width, height, img_width, img_height):
        """
        Convierte bounding box a polígono rectangular

        Args:
            x_center, y_center, width, height: Coordenadas normalizadas del bbox
            img_width, img_height: Dimensiones de la imagen

        Returns:
            list: Coordenadas del polígono normalizado
        """
        # Calcular esquinas del rectángulo
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # Crear polígono rectangular (4 esquinas)
        polygon = [x1, y1, x2, y1, x2, y2, x1, y2]

        return polygon

    def generate_semantic_segmentation_mask(self, instance_masks, class_ids, img_shape):
        """
        Genera máscara de segmentación semántica desde máscaras de instancia

        Args:
            instance_masks (list): Lista de máscaras de instancia
            class_ids (list): Lista de IDs de clase correspondientes
            img_shape (tuple): Forma de la imagen (height, width)

        Returns:
            np.ndarray: Máscara de segmentación semántica
        """
        semantic_mask = np.zeros(img_shape, dtype=np.uint8)

        for mask, class_id in zip(instance_masks, class_ids):
            # Asignar ID de clase a píxeles de la máscara
            semantic_mask[mask > 0] = class_id

        return semantic_mask

    def extract_instance_info(self, mask, class_id, img_width, img_height):
        """
        Extrae información completa de una instancia

        Args:
            mask (np.ndarray): Máscara binaria
            class_id (int): ID de la clase
            img_width, img_height: Dimensiones de la imagen

        Returns:
            dict: Información de la instancia
        """
        if np.sum(mask) == 0:
            return None

        # Detectar componentes conectados
        labeled_mask, num_components = label_region(mask)

        # Calcular área total
        total_area = np.sum(mask > 0)

        # Calcular bounding box global
        coords = np.where(mask > 0)
        ymin, ymax = coords[0].min(), coords[0].max()
        xmin, xmax = coords[1].min(), coords[1].max()

        # Información básica
        instance_info = {
            'class_id': class_id,
            'area': total_area,
            'bbox': {
                'xmin': int(xmin),
                'ymin': int(ymin),
                'xmax': int(xmax),
                'ymax': int(ymax),
                'width': int(xmax - xmin),
                'height': int(ymax - ymin)
            },
            'bbox_normalized': {
                'x_center': (xmin + xmax) / 2 / img_width,
                'y_center': (ymin + ymax) / 2 / img_height,
                'width': (xmax - xmin) / img_width,
                'height': (ymax - ymin) / img_height
            },
            'num_components': num_components,
            'components': []
        }

        # Información de cada componente conectado
        for component_id in range(1, num_components + 1):
            component_mask = (labeled_mask == component_id).astype(np.uint8) * 255
            component_area = np.sum(component_mask > 0)

            # Contorno del componente
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Perímetro
                perimeter = cv2.arcLength(contours[0], True)

                # Ratio de circularidad (4π*área/perímetro²)
                circularity = 4 * np.pi * component_area / (perimeter * perimeter) if perimeter > 0 else 0

                instance_info['components'].append({
                    'id': component_id,
                    'area': component_area,
                    'perimeter': perimeter,
                    'circularity': circularity,
                    'contour_points': len(contours[0])
                })

        return instance_info

    def validate_yolo_label(self, label_line, max_coords=1000):
        """
        Valida una etiqueta YOLO

        Args:
            label_line (str): Línea de etiqueta YOLO
            max_coords (int): Número máximo de coordenadas permitidas

        Returns:
            tuple: (es_válida, mensaje_error)
        """
        try:
            parts = label_line.strip().split()

            if len(parts) < 3:
                return False, "Muy pocos elementos en la etiqueta"

            # Verificar ID de clase
            class_id = int(parts[0])
            if class_id < 0:
                return False, "ID de clase negativo"

            # Verificar coordenadas
            coords = [float(x) for x in parts[1:]]

            if len(coords) % 2 != 0:
                return False, "Número impar de coordenadas"

            if len(coords) > max_coords:
                return False, f"Demasiadas coordenadas ({len(coords)} > {max_coords})"

            # Verificar rango de coordenadas normalizadas
            for coord in coords:
                if coord < 0 or coord > 1:
                    return False, f"Coordenada fuera de rango [0,1]: {coord}"

            return True, "Etiqueta válida"

        except ValueError as e:
            return False, f"Error de formato: {e}"
        except Exception as e:
            return False, f"Error desconocido: {e}"

    def optimize_polygon_points(self, polygon_coords, tolerance=0.01):
        """
        Optimiza un polígono reduciendo puntos redundantes

        Args:
            polygon_coords (list): Coordenadas del polígono [x1,y1,x2,y2,...]
            tolerance (float): Tolerancia para simplificación

        Returns:
            list: Coordenadas optimizadas
        """
        if len(polygon_coords) < 6:  # Mínimo 3 puntos
            return polygon_coords

        # Convertir a array de puntos
        points = np.array(polygon_coords).reshape(-1, 2)

        # Aplicar simplificación de Douglas-Peucker
        simplified_points = self._douglas_peucker(points, tolerance)

        # Convertir de vuelta a lista plana
        return simplified_points.flatten().tolist()

    def _douglas_peucker(self, points, tolerance):
        """
        Implementación del algoritmo Douglas-Peucker para simplificación de polígonos
        """
        if len(points) <= 2:
            return points

        # Encontrar el punto con mayor distancia a la línea entre primer y último punto
        max_distance = 0
        max_index = 0

        for i in range(1, len(points) - 1):
            distance = self._point_to_line_distance(points[i], points[0], points[-1])
            if distance > max_distance:
                max_distance = distance
                max_index = i

        # Si la distancia máxima es mayor que la tolerancia, recursivamente simplificar
        if max_distance > tolerance:
            # Simplificar recursivamente las dos mitades
            left_simplified = self._douglas_peucker(points[:max_index + 1], tolerance)
            right_simplified = self._douglas_peucker(points[max_index:], tolerance)

            # Combinar resultados (evitar duplicar el punto medio)
            return np.vstack([left_simplified[:-1], right_simplified])
        else:
            # Si la distancia es pequeña, devolver solo los puntos extremos
            return np.array([points[0], points[-1]])

    def _point_to_line_distance(self, point, line_start, line_end):
        """
        Calcula la distancia perpendicular de un punto a una línea
        """
        if np.array_equal(line_start, line_end):
            return np.linalg.norm(point - line_start)

        # Vector de la línea
        line_vec = line_end - line_start
        # Vector del punto
        point_vec = point - line_start

        # Proyección del punto sobre la línea
        line_len_sq = np.dot(line_vec, line_vec)
        projection = np.dot(point_vec, line_vec) / line_len_sq

        # Punto más cercano en la línea
        if projection < 0:
            closest_point = line_start
        elif projection > 1:
            closest_point = line_end
        else:
            closest_point = line_start + projection * line_vec

        # Distancia del punto al punto más cercano en la línea
        return np.linalg.norm(point - closest_point)

    def batch_convert_labels(self, input_folder, output_folder, conversion_type='polygon_to_bbox'):
        """
        Convierte etiquetas en lote

        Args:
            input_folder (str): Carpeta con etiquetas originales
            output_folder (str): Carpeta de salida
            conversion_type (str): Tipo de conversión
        """
        import os
        from pathlib import Path

        os.makedirs(output_folder, exist_ok=True)

        label_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

        for label_file in label_files:
            input_path = os.path.join(input_folder, label_file)
            output_path = os.path.join(output_folder, label_file)

            try:
                with open(input_path, 'r') as f:
                    lines = f.readlines()

                converted_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    if conversion_type == 'polygon_to_bbox':
                        converted_line = self._convert_polygon_line_to_bbox(line)
                    else:
                        converted_line = line  # Sin conversión

                    if converted_line:
                        converted_lines.append(converted_line)

                with open(output_path, 'w') as f:
                    for line in converted_lines:
                        f.write(line + '\n')

            except Exception as e:
                print(f"Error convirtiendo {label_file}: {e}")

    def _convert_polygon_line_to_bbox(self, line, img_width=1, img_height=1):
        """
        Convierte una línea de polígono a bbox (asume coordenadas normalizadas)
        """
        try:
            parts = line.split()
            class_id = parts[0]
            coords = [float(x) for x in parts[1:]]

            if len(coords) < 4:
                return None

            # Extraer coordenadas x e y
            x_coords = coords[0::2]
            y_coords = coords[1::2]

            # Calcular bounding box
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)

            # Calcular centro y dimensiones
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

        except Exception as e:
            print(f"Error convirtiendo línea: {e}")
            return None