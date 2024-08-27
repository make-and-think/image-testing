import os
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
from ImageCorruptor import ImageCorruptor
import pandas as pd
from Interrogator import Interrogator

def process_and_collect_stats(input_folder, REPO_ID, general_thresh, character_thresh):
    interrogator = Interrogator()
    interrogator.load_model(REPO_ID)
    corruptor_soft = ImageCorruptor(mode='soft')
    corruptor_hard = ImageCorruptor(mode='hard')

    stats = []

    # Получение списка всех изображений в папке
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        
        # Предсказание для оригинального изображения
        original_ratings, _, _ = interrogator.predict(input_path, general_thresh, character_thresh)

        # Обработка и предсказание для мягкой порчи
        with Image.open(input_path) as img:
            soft_corrupted = corruptor_soft.apply_multiple_modifications(img)
            soft_corrupted_io = io.BytesIO()
            soft_corrupted.save(soft_corrupted_io, format='PNG')
            soft_corrupted_io.seek(0)
            soft_ratings, _, _ = interrogator.predict(soft_corrupted_io, general_thresh, character_thresh)

        # Обработка и предсказание для жёсткой порчи
        with Image.open(input_path) as img:
            hard_corrupted = corruptor_hard.apply_multiple_modifications(img)
            hard_corrupted_io = io.BytesIO()
            hard_corrupted.save(hard_corrupted_io, format='PNG')
            hard_corrupted_io.seek(0)
            hard_ratings, _, _ = interrogator.predict(hard_corrupted_io, general_thresh, character_thresh)

        # Сохранение результатов предсказаний в словарь
        row = {'image': image_file}
        for prefix, ratings in [('original', original_ratings), ('soft', soft_ratings), ('hard', hard_ratings)]:
            for name, value in ratings:
                row[f'{prefix}_{name}'] = value

        stats.append(row)

    return stats

SWINV2_MODEL_DSV3_REPO = "SmilingWolf/wd-swinv2-tagger-v3"

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    input_folder = "D:\\PhotosAndVideos\\Datasets\\GalagenShit"
    #input_folder = os.path.join(current_dir, 'dataset')
    output_folder = os.path.join(current_dir, 'output')
    general_thresh = 0.5
    character_thresh = 0.5

    stats = process_and_collect_stats(input_folder, SWINV2_MODEL_DSV3_REPO, general_thresh, character_thresh)

    # Save statistics to CSV
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df = pd.DataFrame(stats)
    df.to_csv(os.path.join(output_folder, 'statistics.csv'), index=False)

    print("Processing complete. Statistics saved to statistics.csv")


if __name__ == "__main__":
    main()