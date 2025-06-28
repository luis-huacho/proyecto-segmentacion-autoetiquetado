"""
Análisis especializado para dataset de avocados/paltas
Incluye métricas específicas y visualizaciones adaptadas
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import cv2
import os
from collections import defaultdict, Counter
import pandas as pd
from PIL import Image
import json


class AvocadoAnalytics:
    """Clase especializada para análisis de datasets de avocados/paltas"""

    def __init__(self, output_folder):
        """
        Args:
            output_folder (str): Carpeta base de salida
        """
        self.output_folder = output_folder
        self.analytics_folder = os.path.join(output_folder, 'analytics')
        os.makedirs(self.analytics_folder, exist_ok=True)

        # Configurar estilo de visualización
        plt.style.use('default')
        sns.set_palette("husl")

        # Colores específicos para avocados
        self.avocado_colors = {
            'ripe': '#2E4016',  # Verde oscuro maduro
            'unripe': '#7BA428',  # Verde claro inmaduro
            'overripe': '#8B4513',  # Marrón sobre-maduro
            'leaf': '#228B22',  # Verde hoja
            'stem': '#8B4513',  # Marrón tallo
            'branch': '#A0522D',  # Marrón rama
            'flower': '#FFFACD',  # Amarillo claro flor
            'background': '#F5F5DC',  # Beige fondo
            'others': '#D3D3D3'  # Gris otros
        }

        # Métricas acumulativas
        self.metrics_data = {
            'processing_times': [],
            'class_distribution': defaultdict(int),
            'size_distribution': defaultdict(list),
            'quality_metrics': defaultdict(list),
            'spatial_distribution': [],
            'temporal_metrics': []
        }

    def analyze_avocado_detection(self, image_path, detections, masks=None):
        """
        Analiza detecciones específicas de avocados

        Args:
            image_path (str): Ruta de la imagen
            detections (list): Lista de detecciones
            masks (list): Lista de máscaras (opcional)

        Returns:
            dict: Métricas de análisis
        """

        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        if image is None:
            return None

        height, width = image.shape[:2]

        analysis = {
            'image_name': image_name,
            'image_size': (width, height),
            'total_detections': len(detections),
            'avocado_count': 0,
            'maturity_distribution': defaultdict(int),
            'size_analysis': {},
            'density_metrics': {},
            'quality_indicators': {}
        }

        # Analizar cada detección
        avocado_detections = []
        for detection in detections:
            label = detection['label']

            # Filtrar solo avocados (excluir hojas, tallos, etc.)
            if label in ['ripe', 'unripe', 'overripe']:
                avocado_detections.append(detection)
                analysis['avocado_count'] += 1
                analysis['maturity_distribution'][label] += 1

                # Calcular métricas de tamaño
                bbox_area = (detection['xmax'] - detection['xmin']) * (detection['ymax'] - detection['ymin'])
                relative_area = bbox_area / (width * height)

                self.metrics_data['size_distribution'][label].append(relative_area)

        # Análisis de densidad
        if avocado_detections:
            analysis['density_metrics'] = self._calculate_density_metrics(
                avocado_detections, width, height
            )

            # Análisis de distribución espacial
            analysis['spatial_analysis'] = self._analyze_spatial_distribution(
                avocado_detections, width, height
            )

            # Estimación de calidad del cultivo
            analysis['quality_indicators'] = self._estimate_crop_quality(
                avocado_detections, analysis['maturity_distribution']
            )

        # Actualizar métricas globales
        for label, count in analysis['maturity_distribution'].items():
            self.metrics_data['class_distribution'][label] += count

        return analysis

    def _calculate_density_metrics(self, detections, width, height):
        """Calcula métricas de densidad de avocados"""
        total_area = width * height

        # Área total ocupada por avocados
        total_avocado_area = 0
        for det in detections:
            bbox_area = (det['xmax'] - det['xmin']) * (det['ymax'] - det['ymin'])
            total_avocado_area += bbox_area

        density_metrics = {
            'avocados_per_m2': len(detections),  # Asumir escala estándar
            'coverage_percentage': (total_avocado_area / total_area) * 100,
            'avg_avocado_size': total_avocado_area / len(detections) if detections else 0,
            'density_category': 'low'
        }

        # Categorizar densidad
        coverage = density_metrics['coverage_percentage']
        if coverage > 30:
            density_metrics['density_category'] = 'high'
        elif coverage > 15:
            density_metrics['density_category'] = 'medium'

        return density_metrics

    def _analyze_spatial_distribution(self, detections, width, height):
        """Analiza distribución espacial de avocados"""
        if not detections:
            return {}

        # Dividir imagen en grid 3x3
        grid_counts = [[0 for _ in range(3)] for _ in range(3)]

        for det in detections:
            center_x = (det['xmin'] + det['xmax']) / 2
            center_y = (det['ymin'] + det['ymax']) / 2

            # Determinar posición en grid
            grid_x = min(2, int((center_x / width) * 3))
            grid_y = min(2, int((center_y / height) * 3))

            grid_counts[grid_y][grid_x] += 1

        # Calcular métricas de distribución
        flat_counts = [count for row in grid_counts for count in row]

        return {
            'grid_distribution': grid_counts,
            'distribution_uniformity': np.std(flat_counts),
            'max_concentration_area': max(flat_counts),
            'empty_areas': flat_counts.count(0)
        }

    def _estimate_crop_quality(self, detections, maturity_dist):
        """Estima calidad del cultivo basado en distribución de madurez"""
        total_avocados = len(detections)
        if total_avocados == 0:
            return {}

        ripe_pct = (maturity_dist.get('ripe', 0) / total_avocados) * 100
        unripe_pct = (maturity_dist.get('unripe', 0) / total_avocados) * 100
        overripe_pct = (maturity_dist.get('overripe', 0) / total_avocados) * 100

        # Calcular índice de calidad (0-100)
        quality_score = (ripe_pct * 1.0) + (unripe_pct * 0.7) + (overripe_pct * 0.3)

        quality_category = 'poor'
        if quality_score > 80:
            quality_category = 'excellent'
        elif quality_score > 60:
            quality_category = 'good'
        elif quality_score > 40:
            quality_category = 'fair'

        return {
            'quality_score': quality_score,
            'quality_category': quality_category,
            'ripe_percentage': ripe_pct,
            'unripe_percentage': unripe_pct,
            'overripe_percentage': overripe_pct,
            'optimal_harvest_ratio': ripe_pct + (unripe_pct * 0.5)  # Incluir algunos inmaduros
        }

    def create_maturity_distribution_chart(self, save_path=None):
        """Crea gráfico de distribución de madurez de avocados"""
        if not self.metrics_data['class_distribution']:
            return None

        # Filtrar solo categorías de avocados
        avocado_classes = ['ripe', 'unripe', 'overripe']
        avocado_data = {k: v for k, v in self.metrics_data['class_distribution'].items()
                        if k in avocado_classes}

        if not avocado_data:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Gráfico de barras
        classes = list(avocado_data.keys())
        counts = list(avocado_data.values())
        colors = [self.avocado_colors.get(cls, '#gray') for cls in classes]

        bars = ax1.bar(classes, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('Distribución de Madurez de Avocados', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Número de Avocados')
        ax1.set_xlabel('Estado de Madurez')

        # Agregar valores sobre las barras
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                     f'{count}', ha='center', va='bottom', fontweight='bold')

        # Gráfico circular
        ax2.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 12})
        ax2.set_title('Proporción de Madurez', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.analytics_folder, 'maturity_distribution.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def create_size_analysis_chart(self, save_path=None):
        """Crea análisis de distribución de tamaños"""
        size_data = self.metrics_data['size_distribution']

        if not any(size_data.values()):
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Histograma de tamaños por clase
        ax1 = axes[0, 0]
        for class_name, sizes in size_data.items():
            if sizes and class_name in ['ripe', 'unripe', 'overripe']:
                ax1.hist(sizes, alpha=0.6, label=class_name,
                         color=self.avocado_colors.get(class_name), bins=20)

        ax1.set_title('Distribución de Tamaños por Madurez')
        ax1.set_xlabel('Área Relativa (% de imagen)')
        ax1.set_ylabel('Frecuencia')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Box plot de tamaños
        ax2 = axes[0, 1]
        box_data = []
        box_labels = []
        box_colors = []

        for class_name, sizes in size_data.items():
            if sizes and class_name in ['ripe', 'unripe', 'overripe']:
                box_data.append(sizes)
                box_labels.append(class_name)
                box_colors.append(self.avocado_colors.get(class_name))

        if box_data:
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax2.set_title('Distribución de Tamaños (Box Plot)')
        ax2.set_ylabel('Área Relativa (% de imagen)')
        ax2.grid(True, alpha=0.3)

        # 3. Estadísticas de tamaño
        ax3 = axes[1, 0]
        stats_data = []
        for class_name, sizes in size_data.items():
            if sizes and class_name in ['ripe', 'unripe', 'overripe']:
                stats = {
                    'Clase': class_name,
                    'Promedio': np.mean(sizes),
                    'Mediana': np.median(sizes),
                    'Std Dev': np.std(sizes),
                    'Min': np.min(sizes),
                    'Max': np.max(sizes)
                }
                stats_data.append(stats)

        if stats_data:
            df_stats = pd.DataFrame(stats_data)
            table = ax3.table(cellText=df_stats.round(4).values,
                              colLabels=df_stats.columns,
                              cellLoc='center',
                              loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax3.axis('off')
            ax3.set_title('Estadísticas de Tamaño')

        # 4. Correlación tamaño-madurez
        ax4 = axes[1, 1]
        all_sizes = []
        all_labels = []

        for class_name, sizes in size_data.items():
            if sizes and class_name in ['ripe', 'unripe', 'overripe']:
                all_sizes.extend(sizes)
                all_labels.extend([class_name] * len(sizes))

        if all_sizes:
            # Crear scatter plot
            class_to_num = {'unripe': 0, 'ripe': 1, 'overripe': 2}
            x_vals = [class_to_num.get(label, 0) for label in all_labels]

            scatter = ax4.scatter(x_vals, all_sizes,
                                  c=[self.avocado_colors.get(label, 'gray') for label in all_labels],
                                  alpha=0.6, s=50)

            ax4.set_xticks([0, 1, 2])
            ax4.set_xticklabels(['Inmaduro', 'Maduro', 'Sobre-maduro'])
            ax4.set_ylabel('Área Relativa')
            ax4.set_title('Tamaño vs Estado de Madurez')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.analytics_folder, 'size_analysis.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def create_crop_quality_dashboard(self, analysis_results, save_path=None):
        """Crea dashboard de calidad del cultivo"""
        if not analysis_results:
            return None

        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # Recopilar datos de calidad
        quality_scores = []
        density_data = []
        spatial_uniformity = []

        for result in analysis_results:
            if 'quality_indicators' in result:
                qi = result['quality_indicators']
                if 'quality_score' in qi:
                    quality_scores.append(qi['quality_score'])

            if 'density_metrics' in result:
                dm = result['density_metrics']
                density_data.append({
                    'coverage': dm.get('coverage_percentage', 0),
                    'avg_size': dm.get('avg_avocado_size', 0)
                })

            if 'spatial_analysis' in result:
                sa = result['spatial_analysis']
                if 'distribution_uniformity' in sa:
                    spatial_uniformity.append(sa['distribution_uniformity'])

        # 1. Distribución de calidad por imagen
        ax1 = fig.add_subplot(gs[0, :2])
        if quality_scores:
            ax1.hist(quality_scores, bins=20, color='green', alpha=0.7, edgecolor='black')
            ax1.axvline(np.mean(quality_scores), color='red', linestyle='--',
                        label=f'Promedio: {np.mean(quality_scores):.1f}')
            ax1.set_title('Distribución de Puntuación de Calidad', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Puntuación de Calidad (0-100)')
            ax1.set_ylabel('Número de Imágenes')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. Matriz de densidad vs calidad
        ax2 = fig.add_subplot(gs[0, 2:])
        if density_data and quality_scores:
            coverage_vals = [d['coverage'] for d in density_data]
            ax2.scatter(coverage_vals, quality_scores, alpha=0.6, s=60, color='blue')
            ax2.set_xlabel('Cobertura de Avocados (%)')
            ax2.set_ylabel('Puntuación de Calidad')
            ax2.set_title('Densidad vs Calidad del Cultivo')
            ax2.grid(True, alpha=0.3)

            # Línea de tendencia
            if len(coverage_vals) > 1:
                z = np.polyfit(coverage_vals, quality_scores, 1)
                p = np.poly1d(z)
                ax2.plot(sorted(coverage_vals), p(sorted(coverage_vals)), "r--", alpha=0.8)

        # 3. Mapa de calor de distribución espacial promedio
        ax3 = fig.add_subplot(gs[1, :2])
        avg_spatial_grid = np.zeros((3, 3))
        grid_count = 0

        for result in analysis_results:
            if 'spatial_analysis' in result and 'grid_distribution' in result['spatial_analysis']:
                grid = result['spatial_analysis']['grid_distribution']
                avg_spatial_grid += np.array(grid)
                grid_count += 1

        if grid_count > 0:
            avg_spatial_grid /= grid_count
            im = ax3.imshow(avg_spatial_grid, cmap='YlOrRd', aspect='equal')
            ax3.set_title('Distribución Espacial Promedio')
            ax3.set_xticks([0, 1, 2])
            ax3.set_yticks([0, 1, 2])
            ax3.set_xticklabels(['Izquierda', 'Centro', 'Derecha'])
            ax3.set_yticklabels(['Superior', 'Centro', 'Inferior'])

            # Agregar valores en cada celda
            for i in range(3):
                for j in range(3):
                    text = ax3.text(j, i, f'{avg_spatial_grid[i, j]:.1f}',
                                    ha="center", va="center", color="black", fontweight='bold')

            plt.colorbar(im, ax=ax3, label='Densidad Promedio')

        # 4. Métricas de rendimiento del cultivo
        ax4 = fig.add_subplot(gs[1, 2:])

        # Calcular métricas agregadas
        total_avocados = sum(self.metrics_data['class_distribution'][cls]
                             for cls in ['ripe', 'unripe', 'overripe'])

        metrics_text = []
        if total_avocados > 0:
            ripe_pct = (self.metrics_data['class_distribution']['ripe'] / total_avocados) * 100
            unripe_pct = (self.metrics_data['class_distribution']['unripe'] / total_avocados) * 100
            overripe_pct = (self.metrics_data['class_distribution']['overripe'] / total_avocados) * 100

            metrics_text = [
                f"Total de Avocados Detectados: {total_avocados}",
                f"Avocados Maduros: {ripe_pct:.1f}%",
                f"Avocados Inmaduros: {unripe_pct:.1f}%",
                f"Avocados Sobre-maduros: {overripe_pct:.1f}%",
                "",
                f"Calidad Promedio: {np.mean(quality_scores):.1f}/100" if quality_scores else "",
                f"Uniformidad Espacial: {np.mean(spatial_uniformity):.2f}" if spatial_uniformity else "",
                f"Cobertura Promedio: {np.mean([d['coverage'] for d in density_data]):.1f}%" if density_data else ""
            ]

        ax4.text(0.05, 0.95, '\n'.join(metrics_text), transform=ax4.transAxes,
                 fontsize=12, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Métricas del Cultivo', fontweight='bold')

        # 5. Timeline de procesamiento
        ax5 = fig.add_subplot(gs[2, :])
        if hasattr(self, 'processing_timeline') and self.processing_timeline:
            times = [t['timestamp'] for t in self.processing_timeline]
            detections = [t['detections'] for t in self.processing_timeline]

            ax5.plot(times, detections, 'b-', marker='o', markersize=4)
            ax5.set_title('Timeline de Detecciones')
            ax5.set_xlabel('Tiempo de Procesamiento')
            ax5.set_ylabel('Avocados Detectados')
            ax5.grid(True, alpha=0.3)

        # 6. Recomendaciones de cosecha
        ax6 = fig.add_subplot(gs[3, :])

        recommendations = self._generate_harvest_recommendations(analysis_results)
        rec_text = '\n'.join(recommendations)

        ax6.text(0.05, 0.95, rec_text, transform=ax6.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Recomendaciones de Cosecha', fontweight='bold')

        if save_path is None:
            save_path = os.path.join(self.analytics_folder, 'crop_quality_dashboard.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def _generate_harvest_recommendations(self, analysis_results):
        """Genera recomendaciones de cosecha basadas en el análisis"""
        total_avocados = sum(self.metrics_data['class_distribution'][cls]
                             for cls in ['ripe', 'unripe', 'overripe'])

        if total_avocados == 0:
            return ["No se detectaron avocados en las imágenes analizadas."]

        ripe_pct = (self.metrics_data['class_distribution']['ripe'] / total_avocados) * 100
        unripe_pct = (self.metrics_data['class_distribution']['unripe'] / total_avocados) * 100
        overripe_pct = (self.metrics_data['class_distribution']['overripe'] / total_avocados) * 100

        recommendations = []

        # Recomendación principal
        if ripe_pct > 60:
            recommendations.append("🟢 RECOMENDACIÓN: Iniciar cosecha inmediatamente")
            recommendations.append(f"   - Alto porcentaje de frutos maduros ({ripe_pct:.1f}%)")
        elif ripe_pct > 30:
            recommendations.append("🟡 RECOMENDACIÓN: Cosecha selectiva de frutos maduros")
            recommendations.append(f"   - Porcentaje moderado de madurez ({ripe_pct:.1f}%)")
        else:
            recommendations.append("🔴 RECOMENDACIÓN: Esperar 1-2 semanas más")
            recommendations.append(f"   - Bajo porcentaje de madurez ({ripe_pct:.1f}%)")

        # Recomendaciones específicas
        if overripe_pct > 20:
            recommendations.append("⚠️  URGENTE: Cosechar frutos sobre-maduros primero")

        if unripe_pct > 70:
            recommendations.append("📅 PLANIFICACIÓN: Programar cosecha en 2-3 semanas")

        # Recomendaciones de calidad
        avg_quality = np.mean([r.get('quality_indicators', {}).get('quality_score', 0)
                               for r in analysis_results])

        if avg_quality > 80:
            recommendations.append("⭐ CALIDAD: Excelente - apto para mercado premium")
        elif avg_quality > 60:
            recommendations.append("✅ CALIDAD: Buena - apto para mercado estándar")
        else:
            recommendations.append("⚠️  CALIDAD: Revisar técnicas de cultivo")

        return recommendations

    def export_analytics_report(self, analysis_results):
        """Exporta reporte completo en JSON"""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'summary': {
                'total_images_analyzed': len(analysis_results),
                'total_avocados_detected': sum(self.metrics_data['class_distribution'][cls]
                                               for cls in ['ripe', 'unripe', 'overripe']),
                'class_distribution': dict(self.metrics_data['class_distribution']),
                'average_quality_score': np.mean([r.get('quality_indicators', {}).get('quality_score', 0)
                                                  for r in analysis_results]) if analysis_results else 0
            },
            'detailed_analysis': analysis_results,
            'recommendations': self._generate_harvest_recommendations(analysis_results)
        }

        report_path = os.path.join(self.analytics_folder, 'avocado_analysis_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        return report_path

    def add_processing_timeline_entry(self, timestamp, detections_count, image_name):
        """Agrega entrada al timeline de procesamiento"""
        if not hasattr(self, 'processing_timeline'):
            self.processing_timeline = []

        self.processing_timeline.append({
            'timestamp': timestamp,
            'detections': detections_count,
            'image': image_name
        })