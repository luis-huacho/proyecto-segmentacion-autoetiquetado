"""
Utilidades para procesamiento con OpenCLIP
"""

import cv2
import numpy as np
import torch
from PIL import Image
from open_clip import tokenizer


class CLIPProcessor:
    """Clase para procesar imágenes y texto con CLIP"""

    def __init__(self):
        pass

    def apply_mask_to_image(self, image, mask):
        """
        Aplica una máscara a una imagen, manteniendo el color en el área enmascarada
        y estableciendo el resto a blanco

        Args:
            image (np.ndarray): Imagen original
            mask (np.ndarray): Máscara binaria

        Returns:
            np.ndarray: Imagen enmascarada
        """
        # Asegurar que la máscara es binaria
        mask = (mask > 0).astype(np.uint8) * 255

        # Aplicar máscara
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Establecer áreas no enmascaradas a blanco
        masked_image[mask == 0] = 255

        return masked_image

    def crop_object_from_background(self, image):
        """
        Recorta un objeto de un fondo blanco al bounding box mínimo

        Args:
            image (np.ndarray): Imagen con fondo blanco

        Returns:
            tuple: (imagen_recortada, xmin, ymin, xmax, ymax)
        """
        # Convertir a PIL Image para procesamiento
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Convertir a array numpy para análisis
        img_array = np.array(pil_image)

        # Encontrar píxeles no blancos
        non_white_mask = np.any(img_array != 255, axis=2)

        # Encontrar coordenadas del bounding box
        coords = np.where(non_white_mask)
        if len(coords[0]) == 0:
            # Si no hay píxeles no blancos, retornar imagen original
            return pil_image, 0, 0, img_array.shape[1], img_array.shape[0]

        ymin, ymax = coords[0].min(), coords[0].max() + 1
        xmin, xmax = coords[1].min(), coords[1].max() + 1

        # Recortar imagen
        cropped_image = pil_image.crop((xmin, ymin, xmax, ymax))

        return cropped_image, xmin, ymin, xmax, ymax

    def predict_with_clip(self, model, image_input, texts, labels):
        """
        Realiza predicción usando CLIP

        Args:
            model: Modelo CLIP
            image_input (torch.Tensor): Imagen preprocesada
            texts (list): Lista de descripciones
            labels (list): Lista de etiquetas correspondientes

        Returns:
            str: Etiqueta predicha
        """
        # Tokenizar texto con prefijo "This is"
        text_tokens = tokenizer.tokenize(["This is " + desc for desc in texts])
        
        # Mover tokens al dispositivo del modelo
        device = next(model.parameters()).device
        text_tokens = text_tokens.to(device)

        with torch.no_grad():
            # Obtener características de imagen y texto
            image_features = model.encode_image(image_input).float()
            text_features = model.encode_text(text_tokens).float()

            # Normalizar características
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calcular similitud
            similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

            # Obtener etiqueta con mayor similitud
            best_match_idx = np.argmax(similarity)
            predicted_label = labels[best_match_idx]

            return predicted_label

    def get_similarity_scores(self, model, image_input, texts):
        """
        Obtiene scores de similitud para todas las descripciones

        Args:
            model: Modelo CLIP
            image_input (torch.Tensor): Imagen preprocesada
            texts (list): Lista de descripciones

        Returns:
            np.ndarray: Array de scores de similitud
        """
        # Tokenizar texto
        text_tokens = tokenizer.tokenize(["This is " + desc for desc in texts])

        with torch.no_grad():
            # Obtener características
            image_features = model.encode_image(image_input).float()
            text_features = model.encode_text(text_tokens).float()

            # Normalizar
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calcular similitud
            similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

            return similarity.flatten()

    def batch_predict(self, model, image_batch, texts, labels):
        """
        Realiza predicción en lote

        Args:
            model: Modelo CLIP
            image_batch (torch.Tensor): Lote de imágenes
            texts (list): Lista de descripciones
            labels (list): Lista de etiquetas

        Returns:
            list: Lista de etiquetas predichas
        """
        # Tokenizar texto
        text_tokens = tokenizer.tokenize(["This is " + desc for desc in texts])

        with torch.no_grad():
            # Obtener características
            image_features = model.encode_image(image_batch).float()
            text_features = model.encode_text(text_tokens).float()

            # Normalizar
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calcular similitud para cada imagen
            similarities = text_features.cpu().numpy() @ image_features.cpu().numpy().T

            # Obtener mejores coincidencias
            best_matches = np.argmax(similarities, axis=0)
            predicted_labels = [labels[idx] for idx in best_matches]

            return predicted_labels

    def validate_image_quality(self, image, min_size=(32, 32), max_white_ratio=0.95):
        """
        Valida la calidad de una imagen para clasificación

        Args:
            image (PIL.Image): Imagen a validar
            min_size (tuple): Tamaño mínimo (width, height)
            max_white_ratio (float): Ratio máximo de píxeles blancos

        Returns:
            bool: True si la imagen es válida
        """
        # Verificar tamaño mínimo
        if image.size[0] < min_size[0] or image.size[1] < min_size[1]:
            return False

        # Verificar ratio de píxeles blancos
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            white_pixels = np.all(img_array == 255, axis=2)
        else:
            white_pixels = img_array == 255

        white_ratio = np.sum(white_pixels) / (img_array.shape[0] * img_array.shape[1])

        if white_ratio > max_white_ratio:
            return False

        return True

    def preprocess_for_clip(self, image, preprocessor):
        """
        Preprocesa imagen para CLIP

        Args:
            image (PIL.Image): Imagen a preprocesar
            preprocessor: Preprocesador CLIP

        Returns:
            torch.Tensor: Imagen preprocesada
        """
        try:
            processed_image = preprocessor(image)
            return processed_image
        except Exception as e:
            print(f"Error preprocesando imagen: {e}")
            # Crear imagen en blanco como fallback
            return torch.zeros((3, 224, 224))

    def enhance_low_confidence_predictions(self, model, image_input, texts, labels,
                                           confidence_threshold=0.3):
        """
        Mejora predicciones de baja confianza aplicando técnicas adicionales

        Args:
            model: Modelo CLIP
            image_input (torch.Tensor): Imagen preprocesada
            texts (list): Lista de descripciones
            labels (list): Lista de etiquetas
            confidence_threshold (float): Umbral de confianza

        Returns:
            tuple: (etiqueta_predicha, confianza)
        """
        # Obtener scores de similitud
        similarity_scores = self.get_similarity_scores(model, image_input, texts)

        # Normalizar scores a probabilidades
        exp_scores = np.exp(similarity_scores - np.max(similarity_scores))
        probabilities = exp_scores / np.sum(exp_scores)

        best_idx = np.argmax(probabilities)
        confidence = probabilities[best_idx]

        # Si la confianza es baja, intentar con descripciones alternativas
        if confidence < confidence_threshold:
            # Crear descripciones alternativas sin prefijo
            alt_texts = texts.copy()
            alt_similarity = self.get_similarity_scores(model, image_input, alt_texts)

            # Combinar scores
            combined_scores = 0.7 * similarity_scores + 0.3 * alt_similarity
            best_idx = np.argmax(combined_scores)

            # Recalcular confianza
            exp_combined = np.exp(combined_scores - np.max(combined_scores))
            combined_probs = exp_combined / np.sum(exp_combined)
            confidence = combined_probs[best_idx]

        return labels[best_idx], confidence

    def debug_classification(self, model, image_input, texts, labels, image_path=None):
        """
        Proporciona información de debug para clasificación

        Args:
            model: Modelo CLIP
            image_input (torch.Tensor): Imagen preprocesada
            texts (list): Lista de descripciones
            labels (list): Lista de etiquetas
            image_path (str): Ruta de la imagen (opcional)

        Returns:
            dict: Información de debug
        """
        similarity_scores = self.get_similarity_scores(model, image_input, texts)

        # Crear ranking de similitudes
        ranked_indices = np.argsort(similarity_scores)[::-1]

        debug_info = {
            'image_path': image_path,
            'predicted_label': labels[ranked_indices[0]],
            'top_similarities': [
                {
                    'label': labels[idx],
                    'description': texts[idx],
                    'score': float(similarity_scores[idx])
                }
                for idx in ranked_indices[:5]  # Top 5
            ],
            'confidence_gap': float(similarity_scores[ranked_indices[0]] - similarity_scores[ranked_indices[1]]) if len(
                ranked_indices) > 1 else 0,
            'mean_score': float(np.mean(similarity_scores)),
            'std_score': float(np.std(similarity_scores))
        }

        return debug_info