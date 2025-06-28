"""
Sistema de logging y monitoreo para SDM-D Framework
"""

import logging
import os
import time
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict, deque
import threading
import numpy as np


class SDMLogger:
    """Sistema de logging especializado para SDM-D"""

    def __init__(self, output_folder, log_level=logging.INFO, enable_console=True):
        """
        Inicializa el sistema de logging

        Args:
            output_folder (str): Carpeta base de salida
            log_level: Nivel de logging
            enable_console (bool): Habilitar logging en consola
        """
        self.output_folder = output_folder
        self.log_folder = os.path.join(output_folder, 'logs')
        os.makedirs(self.log_folder, exist_ok=True)

        # Crear timestamp para esta sesión
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Configurar loggers
        self._setup_loggers(log_level, enable_console)

        # Métricas de progreso
        self.progress_data = {
            'start_time': time.time(),
            'current_phase': None,
            'processed_images': 0,
            'total_images': 0,
            'current_image': None,
            'phase_start_time': 0,
            'errors': [],
            'warnings': [],
            'phase_metrics': defaultdict(dict)
        }

        # Buffer para métricas en tiempo real
        self.metrics_buffer = deque(maxlen=1000)

    def _setup_loggers(self, log_level, enable_console):
        """Configura los diferentes loggers"""

        # Logger principal
        self.main_logger = logging.getLogger('sdm_main')
        self.main_logger.setLevel(log_level)

        # Logger de segmentación
        self.seg_logger = logging.getLogger('sdm_segmentation')
        self.seg_logger.setLevel(log_level)

        # Logger de anotación
        self.ann_logger = logging.getLogger('sdm_annotation')
        self.ann_logger.setLevel(log_level)

        # Logger de métricas
        self.metrics_logger = logging.getLogger('sdm_metrics')
        self.metrics_logger.setLevel(log_level)

        # Crear manejadores de archivo
        main_handler = logging.FileHandler(
            os.path.join(self.log_folder, f'main_{self.session_id}.log')
        )
        seg_handler = logging.FileHandler(
            os.path.join(self.log_folder, f'segmentation_{self.session_id}.log')
        )
        ann_handler = logging.FileHandler(
            os.path.join(self.log_folder, f'annotation_{self.session_id}.log')
        )
        metrics_handler = logging.FileHandler(
            os.path.join(self.log_folder, f'metrics_{self.session_id}.log')
        )

        # Formato de logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        main_handler.setFormatter(formatter)
        seg_handler.setFormatter(formatter)
        ann_handler.setFormatter(formatter)
        metrics_handler.setFormatter(formatter)

        # Agregar manejadores
        self.main_logger.addHandler(main_handler)
        self.seg_logger.addHandler(seg_handler)
        self.ann_logger.addHandler(ann_handler)
        self.metrics_logger.addHandler(metrics_handler)

        # Manejador de consola si está habilitado
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))

            self.main_logger.addHandler(console_handler)

    def start_phase(self, phase_name, total_items=0):
        """Inicia una nueva fase de procesamiento"""
        self.progress_data['current_phase'] = phase_name
        self.progress_data['phase_start_time'] = time.time()
        self.progress_data['total_images'] = total_items
        self.progress_data['processed_images'] = 0

        self.main_logger.info(f"🚀 Iniciando fase: {phase_name}")
        if total_items > 0:
            self.main_logger.info(f"📊 Total de elementos a procesar: {total_items}")

    def end_phase(self, phase_name):
        """Finaliza una fase de procesamiento"""
        phase_time = time.time() - self.progress_data['phase_start_time']

        # Guardar métricas de la fase
        self.progress_data['phase_metrics'][phase_name] = {
            'duration': phase_time,
            'processed_items': self.progress_data['processed_images'],
            'avg_time_per_item': phase_time / max(1, self.progress_data['processed_images']),
            'end_time': datetime.now().isoformat()
        }

        self.main_logger.info(f"✅ Fase completada: {phase_name}")
        self.main_logger.info(f"⏱️ Tiempo transcurrido: {phase_time:.2f} segundos")
        self.main_logger.info(f"📊 Elementos procesados: {self.progress_data['processed_images']}")

        if self.progress_data['processed_images'] > 0:
            avg_time = phase_time / self.progress_data['processed_images']
            self.main_logger.info(f"⚡ Tiempo promedio por elemento: {avg_time:.2f} segundos")

    def log_image_processing(self, image_name, phase, processing_time=None,
                             masks_generated=0, detections_made=0, error=None):
        """Registra el procesamiento de una imagen individual"""

        self.progress_data['processed_images'] += 1
        self.progress_data['current_image'] = image_name

        # Calcular progreso
        if self.progress_data['total_images'] > 0:
            progress_pct = (self.progress_data['processed_images'] /
                            self.progress_data['total_images']) * 100
        else:
            progress_pct = 0

        # Log básico
        if error:
            logger = self.main_logger.error
            status = "❌ ERROR"
            self.progress_data['errors'].append({
                'image': image_name,
                'phase': phase,
                'error': str(error),
                'timestamp': datetime.now().isoformat()
            })
        else:
            logger = self.main_logger.info
            status = "✅ OK"

        logger(f"{status} [{progress_pct:.1f}%] {phase}: {image_name}")

        if processing_time:
            logger(f"   ⏱️ Tiempo: {processing_time:.2f}s")
        if masks_generated > 0:
            logger(f"   🎭 Máscaras: {masks_generated}")
        if detections_made > 0:
            logger(f"   🎯 Detecciones: {detections_made}")

        # Agregar métrica al buffer
        metric = {
            'timestamp': time.time(),
            'image': image_name,
            'phase': phase,
            'processing_time': processing_time or 0,
            'masks_generated': masks_generated,
            'detections_made': detections_made,
            'progress_pct': progress_pct,
            'error': error is not None
        }
        self.metrics_buffer.append(metric)

        # Log detallado para métricas
        self.metrics_logger.info(json.dumps(metric))

    def log_segmentation_metrics(self, image_name, num_masks_raw, num_masks_filtered,
                                 nms_applied, processing_time):
        """Registra métricas específicas de segmentación"""

        self.seg_logger.info(f"📊 Segmentación: {image_name}")
        self.seg_logger.info(f"   🎭 Máscaras brutas: {num_masks_raw}")
        self.seg_logger.info(f"   🎯 Máscaras filtradas: {num_masks_filtered}")
        self.seg_logger.info(f"   🔧 NMS aplicado: {'Sí' if nms_applied else 'No'}")
        self.seg_logger.info(f"   ⏱️ Tiempo: {processing_time:.2f}s")

        if num_masks_raw > 0:
            reduction_pct = ((num_masks_raw - num_masks_filtered) / num_masks_raw) * 100
            self.seg_logger.info(f"   📉 Reducción por NMS: {reduction_pct:.1f}%")

    def log_annotation_metrics(self, image_name, num_masks, classification_results,
                               processing_time):
        """Registra métricas específicas de anotación"""

        self.ann_logger.info(f"🏷️ Anotación: {image_name}")
        self.ann_logger.info(f"   🎭 Máscaras procesadas: {num_masks}")
        self.ann_logger.info(f"   ⏱️ Tiempo: {processing_time:.2f}s")

        # Distribución de clases
        if classification_results:
            class_counts = defaultdict(int)
            for result in classification_results:
                class_counts[result.get('label', 'unknown')] += 1

            self.ann_logger.info("   📊 Distribución de clases:")
            for class_name, count in class_counts.items():
                self.ann_logger.info(f"      {class_name}: {count}")

    def log_warning(self, message, component="main"):
        """Registra una advertencia"""
        self.progress_data['warnings'].append({
            'message': message,
            'component': component,
            'timestamp': datetime.now().isoformat()
        })

        if component == "segmentation":
            logger = self.seg_logger
        elif component == "annotation":
            logger = self.ann_logger
        else:
            logger = self.main_logger

        logger.warning(f"⚠️ {message}")

    def log_error(self, error, component="main", context=None):
        """Registra un error"""
        error_info = {
            'error': str(error),
            'component': component,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        self.progress_data['errors'].append(error_info)

        if component == "segmentation":
            logger = self.seg_logger
        elif component == "annotation":
            logger = self.ann_logger
        else:
            logger = self.main_logger

        logger.error(f"❌ {error}")
        if context:
            logger.error(f"   Contexto: {context}")

    def get_progress_summary(self):
        """Obtiene resumen del progreso actual"""
        current_time = time.time()
        elapsed_time = current_time - self.progress_data['start_time']

        summary = {
            'session_id': self.session_id,
            'current_phase': self.progress_data['current_phase'],
            'elapsed_time': elapsed_time,
            'processed_images': self.progress_data['processed_images'],
            'total_images': self.progress_data['total_images'],
            'current_image': self.progress_data['current_image'],
            'errors_count': len(self.progress_data['errors']),
            'warnings_count': len(self.progress_data['warnings']),
            'progress_percentage': 0
        }

        if self.progress_data['total_images'] > 0:
            summary['progress_percentage'] = (
                                                     self.progress_data['processed_images'] /
                                                     self.progress_data['total_images']
                                             ) * 100

        # Calcular velocidad de procesamiento
        if elapsed_time > 0:
            summary['processing_rate'] = self.progress_data['processed_images'] / elapsed_time

            if summary['processing_rate'] > 0 and self.progress_data['total_images'] > 0:
                remaining_images = self.progress_data['total_images'] - self.progress_data['processed_images']
                summary['estimated_time_remaining'] = remaining_images / summary['processing_rate']

        return summary

    def save_session_report(self):
        """Guarda reporte completo de la sesión"""
        report_path = os.path.join(self.log_folder, f'session_report_{self.session_id}.json')

        # Preparar reporte completo
        total_time = time.time() - self.progress_data['start_time']

        report = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': datetime.fromtimestamp(self.progress_data['start_time']).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': total_time,
                'output_folder': self.output_folder
            },
            'processing_summary': {
                'total_images_processed': self.progress_data['processed_images'],
                'total_errors': len(self.progress_data['errors']),
                'total_warnings': len(self.progress_data['warnings']),
                'average_processing_time': total_time / max(1, self.progress_data['processed_images'])
            },
            'phase_metrics': dict(self.progress_data['phase_metrics']),
            'errors': self.progress_data['errors'],
            'warnings': self.progress_data['warnings']
        }

        # Agregar métricas de rendimiento
        if self.metrics_buffer:
            processing_times = [m['processing_time'] for m in self.metrics_buffer if m['processing_time'] > 0]
            if processing_times:
                report['performance_metrics'] = {
                    'min_processing_time': min(processing_times),
                    'max_processing_time': max(processing_times),
                    'avg_processing_time': np.mean(processing_times),
                    'std_processing_time': np.std(processing_times)
                }

        # Guardar reporte
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.main_logger.info(f"📄 Reporte de sesión guardado: {report_path}")
        return report_path

    def create_progress_visualization(self, save_path=None):
        """Crea visualización del progreso en tiempo real"""
        if not self.metrics_buffer:
            return None

        # Preparar datos
        timestamps = [m['timestamp'] for m in self.metrics_buffer]
        processing_times = [m['processing_time'] for m in self.metrics_buffer]
        progress_pcts = [m['progress_pct'] for m in self.metrics_buffer]

        # Convertir timestamps a datetime
        start_time = min(timestamps)
        relative_times = [(t - start_time) / 60 for t in timestamps]  # En minutos

        # Crear gráfica
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Gráfica 1: Progreso vs Tiempo
        ax1.plot(relative_times, progress_pcts, 'b-', linewidth=2, marker='o', markersize=3)
        ax1.set_xlabel('Tiempo (minutos)')
        ax1.set_ylabel('Progreso (%)')
        ax1.set_title('Progreso del Procesamiento')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)

        # Gráfica 2: Tiempo de procesamiento por imagen
        ax2.plot(relative_times, processing_times, 'r-', linewidth=1, alpha=0.7)
        if len(processing_times) > 10:
            # Agregar media móvil
            window_size = min(10, len(processing_times) // 3)
            moving_avg = np.convolve(processing_times, np.ones(window_size) / window_size, mode='valid')
            moving_avg_times = relative_times[window_size - 1:]
            ax2.plot(moving_avg_times, moving_avg, 'g-', linewidth=2, label=f'Media móvil ({window_size})')
            ax2.legend()

        ax2.set_xlabel('Tiempo (minutos)')
        ax2.set_ylabel('Tiempo por imagen (s)')
        ax2.set_title('Velocidad de Procesamiento')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.log_folder, f'progress_{self.session_id}.png')

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.main_logger.info(f"📊 Visualización de progreso guardada: {save_path}")
        return save_path


class ProgressMonitor:
    """Monitor de progreso en tiempo real con actualización automática"""

    def __init__(self, logger, update_interval=30):
        """
        Args:
            logger (SDMLogger): Logger principal
            update_interval (int): Intervalo de actualización en segundos
        """
        self.logger = logger
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Inicia el monitoreo en background"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.main_logger.info("📊 Monitor de progreso iniciado")

    def stop_monitoring(self):
        """Detiene el monitoreo"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.main_logger.info("📊 Monitor de progreso detenido")

    def _monitor_loop(self):
        """Loop principal del monitor"""
        while self.monitoring:
            try:
                # Obtener resumen de progreso
                summary = self.logger.get_progress_summary()

                # Log de estado actual
                self._log_progress_status(summary)

                # Crear visualización actualizada cada 5 minutos
                if hasattr(self, '_last_visualization_time'):
                    if time.time() - self._last_visualization_time > 300:  # 5 minutos
                        self.logger.create_progress_visualization()
                        self._last_visualization_time = time.time()
                else:
                    self._last_visualization_time = time.time()

                time.sleep(self.update_interval)

            except Exception as e:
                self.logger.log_error(f"Error en monitor de progreso: {e}", "monitor")
                time.sleep(10)  # Esperar más tiempo si hay error

    def _log_progress_status(self, summary):
        """Log del estado actual del progreso"""
        phase = summary.get('current_phase', 'N/A')
        progress = summary.get('progress_percentage', 0)
        processed = summary.get('processed_images', 0)
        total = summary.get('total_images', 0)
        current_img = summary.get('current_image', 'N/A')

        status_msg = f"📊 Estado [{phase}]: {progress:.1f}% ({processed}/{total}) - Actual: {current_img}"

        if 'processing_rate' in summary:
            rate = summary['processing_rate']
            status_msg += f" - Velocidad: {rate:.2f} img/s"

        if 'estimated_time_remaining' in summary:
            eta_minutes = summary['estimated_time_remaining'] / 60
            status_msg += f" - ETA: {eta_minutes:.1f} min"

        self.logger.main_logger.info(status_msg)

        # Log errores/warnings si los hay
        if summary.get('errors_count', 0) > 0:
            self.logger.main_logger.warning(f"⚠️ Errores acumulados: {summary['errors_count']}")
        if summary.get('warnings_count', 0) > 0:
            self.logger.main_logger.info(f"💡 Advertencias: {summary['warnings_count']}")