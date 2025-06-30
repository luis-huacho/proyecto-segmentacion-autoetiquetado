import torch
import open_clip
import argparse
import csv
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def parse_descriptions(file_path):
    """
    Lee el archivo de descripciones y lo separa en dos listas:
    una con las frases completas y otra con las etiquetas cortas.
    """
    descriptions = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Dividir solo en la última coma para separar descripción y etiqueta
                parts = line.rsplit(',', 1)
                if len(parts) == 2:
                    descriptions.append(parts[0].strip())
                    labels.append(parts[1].strip())
    return descriptions, labels


def main(args):
    """
    Función principal que procesa las imágenes y genera el CSV.
    """
    # 1. Configuración del dispositivo (GPU si está disponible)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # 2. Cargar el modelo OpenCLIP y el preprocesador de imágenes
    print("Cargando modelo OpenCLIP...")
    # Asegúrate de que el modelo y el pre-entrenamiento coincidan con tu pipeline original
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14',
        pretrained='laion2b_s32b_b79k',
        device=device
    )
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    print("Modelo cargado.")

    # 3. Cargar y procesar las descripciones de texto
    print(f"Cargando descripciones desde: {args.desc_file}")
    descriptions, labels = parse_descriptions(args.desc_file)
    if not descriptions:
        print("Error: No se encontraron descripciones válidas en el archivo.")
        return

    # Tokenizar el texto y moverlo al dispositivo
    text_tokens = tokenizer(descriptions).to(device)

    # 4. Preparar el archivo CSV de salida
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Encontrar todas las imágenes en el directorio de entrada
    dataset_path = Path(args.dataset_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_paths = [p for p in dataset_path.rglob('*') if p.suffix.lower() in image_extensions]

    print(f"Se encontraron {len(image_paths)} imágenes. Procesando...")

    # 5. Iterar sobre las imágenes y generar etiquetas
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_name', 'predicted_label', 'confidence'])

        # Usamos tqdm para una barra de progreso
        for image_path in tqdm(image_paths, desc="Clasificando imágenes"):
            try:
                image = Image.open(image_path).convert("RGB")
                # Preprocesar la imagen y moverla al dispositivo
                image_tensor = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad(), torch.cuda.amp.autocast():
                    # Obtener las características de la imagen y el texto
                    image_features = model.encode_image(image_tensor)
                    text_features = model.encode_text(text_tokens)

                    # Normalizar características para calcular similitud de coseno
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    # Calcular probabilidades de similitud
                    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                    # Encontrar la mejor coincidencia
                    top_prob, top_idx = text_probs.cpu().topk(1)
                    best_label_index = top_idx.item()

                    predicted_label = labels[best_label_index]
                    confidence = top_prob.item()

                    # Escribir en el CSV
                    csv_writer.writerow([image_path.name, predicted_label, f"{confidence:.4f}"])

            except Exception as e:
                print(f"\nError procesando {image_path.name}: {e}")
                # Opcional: escribir una fila de error en el CSV
                csv_writer.writerow([image_path.name, 'ERROR', str(e)])

    print(f"\n¡Proceso completado! Resultados guardados en: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clasifica imágenes usando OpenCLIP y genera un CSV.")
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help="Ruta al directorio principal del dataset (que contiene train, test, val)."
    )
    parser.add_argument(
        '--desc_file',
        type=str,
        default='description/avocado_des.txt',
        help="Ruta al archivo de texto con las descripciones."
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='output/clip_labels.csv',
        help="Ruta donde se guardará el archivo CSV resultante."
    )

    args = parser.parse_args()
    main(args)
