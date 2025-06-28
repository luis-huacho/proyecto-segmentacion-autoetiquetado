"""
Utilidades para visualización de resultados
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class VisualizationManager:
    """Clase para manejar todas las visualizaciones"""

    def __init__(self):
        # Colores predefinidos para diferentes clases
        self.class_colors = {
            'ripe': (229, 76, 94),  # Rojo/Rosa
            'unripe': (146, 208, 80),  # Verde claro
            'leaf': (0, 176, 80),  # Verde
            'stem': (243, 163, 97),  # Naranja
            'flower': (168, 218, 219),  # Azul claro
            'others': (252, 248, 187),  # Amarillo claro
            'background': (128, 128, 128)  # Gris
        }

    def create_box_visualization(self, image_path, detections, output_path):
        """
        Crea visualización con cajas delimitadoras

        Args:
            image_path (str): Ruta de la imagen original
            detections (list): Lista de detecciones con bounding boxes
            output_path (str): Ruta de salida
        """
        try:
            # Cargar imagen
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ No se pudo cargar imagen: {image_path}")
                return

            # Configuración de texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3
            box_thickness = 7

            for detection in detections:
                label = detection['label']

                # Saltar 'others' si no queremos visualizarlo
                if label == 'others':
                    continue

                # Obtener coordenadas y color
                xmin, ymin = detection['xmin'], detection['ymin']
                xmax, ymax = detection['xmax'], detection['ymax']
                color = self.class_colors.get(label, (76, 94, 229))  # Color por defecto azul

                # Dibujar rectángulo
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, box_thickness)

                # Preparar texto de etiqueta
                label_text = label
                (label_width, label_height), baseline = cv2.getTextSize(
                    label_text, font, font_scale, thickness
                )

                # Dibujar fondo blanco para el texto
                top_left = (xmin, ymin - label_height - baseline)
                bottom_right = (xmin + label_width, ymin - baseline)
                cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), cv2.FILLED)

                # Dibujar texto
                text_origin = (xmin, ymin - baseline)
                cv2.putText(img, label_text, text_origin, font, font_scale, color, thickness)

            # Guardar imagen
            cv2.imwrite(output_path, img)

        except Exception as e:
            print(f"⚠️ Error creando visualización de cajas: {e}")

    def create_color_mask_visualization(self, image, masks, detections, output_path):
        """
        Crea visualización con máscaras coloreadas

        Args:
            image (np.ndarray): Imagen original en RGB
            masks (list): Lista de máscaras anotadas
            detections (list): Lista de detecciones
            output_path (str): Ruta de salida
        """
        try:
            # Crear figura
            fig, ax = plt.subplots(figsize=(20, 20))
            ax.imshow(image)
            ax.set_autoscale_on(False)

            # Crear overlay para máscaras
            img_with_masks = image.copy()
            overlay = np.zeros_like(img_with_masks, dtype=np.uint8)

            # Aplicar colores a las máscaras
            alpha = 0.4
            for mask_info in masks:
                mask = mask_info['segmentation']
                label = mask_info['label']

                if label == 'others':
                    continue  # Saltar 'others'

                color = self.class_colors.get(label, (128, 128, 128))
                overlay[mask > 0] = color

            # Combinar imagen original con overlay
            img_with_masks = cv2.addWeighted(overlay, alpha, img_with_masks, 1 - alpha, 0)

            # Mostrar imagen con máscaras
            ax.imshow(img_with_masks)

            # Dibujar cajas y etiquetas
            for detection in detections:
                label = detection['label']

                if label == 'others':
                    continue

                # Obtener color normalizado para matplotlib
                color = self.class_colors.get(label, (128, 128, 128))
                color_norm = tuple(c / 255.0 for c in color)

                # Dibujar rectángulo
                xmin, ymin = detection['xmin'], detection['ymin']
                width = detection['xmax'] - detection['xmin']
                height = detection['ymax'] - detection['ymin']

                rect = plt.Rectangle((xmin, ymin), width, height,
                                     linewidth=2, edgecolor=color_norm, facecolor='none')
                ax.add_patch(rect)

                # Agregar texto
                ax.text(xmin, ymin - 5, label, color='white', fontsize=30,
                        ha='left', va='bottom',
                        bbox=dict(facecolor=color_norm, edgecolor='none', boxstyle='round,pad=0'))

            # Guardar
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"⚠️ Error creando visualización coloreada: {e}")

    def create_comparison_visualization(self, original_image, segmented_image,
                                        annotated_image, output_path):
        """
        Crea visualización de comparación con 3 paneles

        Args:
            original_image (np.ndarray): Imagen original
            segmented_image (np.ndarray): Imagen con segmentación
            annotated_image (np.ndarray): Imagen con anotaciones
            output_path (str): Ruta de salida
        """
        try:
            fig, axes = plt.subplots(1, 3, figsize=(30, 10))

            # Panel 1: Imagen original
            axes[0].imshow(original_image)
            axes[0].set_title('Original', fontsize=16, fontweight='bold')
            axes[0].axis('off')

            # Panel 2: Segmentación
            axes[1].imshow(segmented_image)
            axes[1].set_title('Segmentación SAM2', fontsize=16, fontweight='bold')
            axes[1].axis('off')

            # Panel 3: Anotaciones
            axes[2].imshow(annotated_image)
            axes[2].set_title('Anotaciones CLIP', fontsize=16, fontweight='bold')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"⚠️ Error creando visualización de comparación: {e}")

    def create_statistics_visualization(self, stats_data, output_path):
        """
        Crea visualización de estadísticas del procesamiento

        Args:
            stats_data (dict): Datos estadísticos
            output_path (str): Ruta de salida
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Estadísticas del Procesamiento SDM-D', fontsize=16, fontweight='bold')

            # Gráfico 1: Distribución de clases
            if 'class_distribution' in stats_data:
                labels = list(stats_data['class_distribution'].keys())
                sizes = list(stats_data['class_distribution'].values())
                colors = [self._get_color_for_class(label) for label in labels]

                axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
                axes[0, 0].set_title('Distribución de Clases')

            # Gráfico 2: Máscaras por imagen
            if 'masks_per_image' in stats_data:
                axes[0, 1].hist(stats_data['masks_per_image'], bins=20, alpha=0.7, color='skyblue')
                axes[0, 1].set_title('Máscaras por Imagen')
                axes[0, 1].set_xlabel('Número de Máscaras')
                axes[0, 1].set_ylabel('Frecuencia')

            # Gráfico 3: Tamaños de área
            if 'area_sizes' in stats_data:
                axes[1, 0].hist(stats_data['area_sizes'], bins=30, alpha=0.7, color='lightgreen')
                axes[1, 0].set_title('Distribución de Tamaños de Área')
                axes[1, 0].set_xlabel('Área (píxeles)')
                axes[1, 0].set_ylabel('Frecuencia')
                axes[1, 0].set_yscale('log')

            # Gráfico 4: Scores de confianza
            if 'confidence_scores' in stats_data:
                axes[1, 1].hist(stats_data['confidence_scores'], bins=20, alpha=0.7, color='orange')
                axes[1, 1].set_title('Distribución de Scores de Confianza')
                axes[1, 1].set_xlabel('Score de Confianza')
                axes[1, 1].set_ylabel('Frecuencia')

            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"⚠️ Error creando visualización de estadísticas: {e}")

    def create_detection_grid(self, image_paths, detection_lists, output_path,
                              grid_size=(2, 3), figsize=(20, 15)):
        """
        Crea una grilla de detecciones para múltiples imágenes

        Args:
            image_paths (list): Lista de rutas de imágenes
            detection_lists (list): Lista de listas de detecciones
            output_path (str): Ruta de salida
            grid_size (tuple): Tamaño de la grilla (filas, columnas)
            figsize (tuple): Tamaño de la figura
        """
        try:
            rows, cols = grid_size
            fig, axes = plt.subplots(rows, cols, figsize=figsize)

            if rows * cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            for i, (image_path, detections) in enumerate(zip(image_paths, detection_lists)):
                if i >= len(axes):
                    break

                # Cargar y mostrar imagen
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Dibujar detecciones
                annotated_image = self._draw_detections_on_image(image_rgb, detections)

                axes[i].imshow(annotated_image)
                axes[i].set_title(f'Imagen {i + 1}: {len(detections)} detecciones',
                                  fontsize=12, fontweight='bold')
                axes[i].axis('off')

            # Ocultar ejes no utilizados
            for i in range(len(image_paths), len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"⚠️ Error creando grilla de detecciones: {e}")

    def _draw_detections_on_image(self, image, detections):
        """
        Dibuja detecciones sobre una imagen

        Args:
            image (np.ndarray): Imagen en formato RGB
            detections (list): Lista de detecciones

        Returns:
            np.ndarray: Imagen con detecciones dibujadas
        """
        annotated_image = image.copy()

        for detection in detections:
            label = detection['label']

            if label == 'others':
                continue

            # Obtener coordenadas y color
            xmin, ymin = detection['xmin'], detection['ymin']
            xmax, ymax = detection['xmax'], detection['ymax']
            color = self.class_colors.get(label, (76, 94, 229))

            # Dibujar rectángulo
            cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), color, 3)

            # Dibujar etiqueta
            cv2.putText(annotated_image, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return annotated_image

    def _get_color_for_class(self, class_name):
        """
        Obtiene color normalizado para matplotlib

        Args:
            class_name (str): Nombre de la clase

        Returns:
            tuple: Color normalizado (0-1)
        """
        color = self.class_colors.get(class_name, (128, 128, 128))
        return tuple(c / 255.0 for c in color)

    def create_confusion_matrix_visualization(self, confusion_matrix, class_names, output_path):
        """
        Crea visualización de matriz de confusión

        Args:
            confusion_matrix (np.ndarray): Matriz de confusión
            class_names (list): Nombres de las clases
            output_path (str): Ruta de salida
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Crear heatmap
            im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)

            # Configurar etiquetas
            ax.set(xticks=np.arange(confusion_matrix.shape[1]),
                   yticks=np.arange(confusion_matrix.shape[0]),
                   xticklabels=class_names,
                   yticklabels=class_names,
                   title='Matriz de Confusión',
                   ylabel='Etiqueta Real',
                   xlabel='Etiqueta Predicha')

            # Rotar etiquetas del eje x
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Agregar texto en cada celda
            thresh = confusion_matrix.max() / 2.
            for i in range(confusion_matrix.shape[0]):
                for j in range(confusion_matrix.shape[1]):
                    ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if confusion_matrix[i, j] > thresh else "black")

            fig.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"⚠️ Error creando matriz de confusión: {e}")

    def create_progress_visualization(self, progress_data, output_path):
        """
        Crea visualización del progreso del procesamiento

        Args:
            progress_data (dict): Datos de progreso
            output_path (str): Ruta de salida
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Gráfico de progreso temporal
            if 'timestamps' in progress_data and 'processed_count' in progress_data:
                ax1.plot(progress_data['timestamps'], progress_data['processed_count'],
                         'b-', linewidth=2, marker='o')
                ax1.set_title('Progreso del Procesamiento')
                ax1.set_xlabel('Tiempo')
                ax1.set_ylabel('Imágenes Procesadas')
                ax1.grid(True, alpha=0.3)

            # Gráfico de velocidad de procesamiento
            if 'processing_rates' in progress_data:
                ax2.bar(range(len(progress_data['processing_rates'])),
                        progress_data['processing_rates'], color='skyblue', alpha=0.7)
                ax2.set_title('Velocidad de Procesamiento')
                ax2.set_xlabel('Lote')
                ax2.set_ylabel('Imágenes/segundo')
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"⚠️ Error creando visualización de progreso: {e}")

    def generate_summary_report(self, stats, output_folder):
        """
        Genera reporte visual completo

        Args:
            stats (dict): Estadísticas completas
            output_folder (str): Carpeta de salida
        """
        import os

        # Crear carpeta para reportes si no existe
        report_folder = os.path.join(output_folder, 'reports')
        os.makedirs(report_folder, exist_ok=True)

        # Generar diferentes visualizaciones
        if 'class_distribution' in stats:
            self.create_statistics_visualization(
                stats,
                os.path.join(report_folder, 'statistics_summary.png')
            )

        if 'sample_images' in stats and 'sample_detections' in stats:
            self.create_detection_grid(
                stats['sample_images'][:6],  # Máximo 6 imágenes
                stats['sample_detections'][:6],
                os.path.join(report_folder, 'detection_samples.png')
            )

        print(f"📊 Reporte visual generado en: {report_folder}")