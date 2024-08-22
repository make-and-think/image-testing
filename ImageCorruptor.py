import os
import random
import numpy as np
import io
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time

class ImageCorruptor:
    def __init__(self, mode='hard'):
        self.mode = mode
        self.modification_functions = {
            'soft': [
                self.soft_noise,
                self.soft_blur,
                self.soft_brightness,
                self.soft_crop,
                self.soft_rotate,
                self.soft_contrast,
                self.soft_random_pixels
            ],
            'hard': [
                self.extreme_noise,
                self.extreme_blur,
                self.extreme_brightness,
                self.extreme_crop,
                self.extreme_rotate,
                self.extreme_contrast,
                self.extreme_random_pixels,
                self.color_inversion,
                self.extreme_compression,
                self.random_channels
            ]
        }

    @staticmethod
    def convert_to_rgb(image):
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            return image.convert('RGB')
        return image

    # Soft modifications
    def soft_noise(self, image):
        image = self.convert_to_rgb(image)
        noise_factor = random.uniform(0.1, 0.3)
        img_array = np.array(image)
        noise = np.random.randn(*img_array.shape) * 255 * noise_factor
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def soft_blur(self, image):
        return image.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.0)))

    def soft_brightness(self, image):
        factor = random.uniform(0.8, 1.2)
        return ImageEnhance.Brightness(image).enhance(factor)

    def soft_crop(self, image):
        factor = random.uniform(0.8, 0.95)
        width, height = image.size
        left = width * (1-factor) / 2
        top = height * (1-factor) / 2
        right = width - left
        bottom = height - top
        return image.crop((left, top, right, bottom))

    def soft_rotate(self, image):
        return image.rotate(random.uniform(-15, 15))

    def soft_contrast(self, image):
        factor = random.uniform(0.8, 1.2)
        return ImageEnhance.Contrast(image).enhance(factor)

    def soft_random_pixels(self, image):
        image = self.convert_to_rgb(image)
        num_pixels = random.randint(100, 1000)
        img_array = np.array(image)
        for _ in range(num_pixels):
            x = np.random.randint(0, img_array.shape[1])
            y = np.random.randint(0, img_array.shape[0])
            img_array[y, x] = np.random.randint(0, 256, size=3)
        return Image.fromarray(img_array)

    # Hard modifications
    def extreme_noise(self, image):
        image = self.convert_to_rgb(image)
        noise_factor = random.uniform(0.5, 1.5)
        img_array = np.array(image)
        noise = np.random.randn(*img_array.shape) * 255 * noise_factor
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def extreme_blur(self, image):
        return image.filter(ImageFilter.GaussianBlur(random.uniform(5.0, 20.0)))

    def extreme_brightness(self, image):
        factor = random.choice([random.uniform(0.1, 0.3), random.uniform(3.0, 5.0)])
        return ImageEnhance.Brightness(image).enhance(factor)

    def extreme_crop(self, image):
        factor = random.uniform(0.3, 0.6)
        width, height = image.size
        left = width * (1-factor) / 2
        top = height * (1-factor) / 2
        right = width - left
        bottom = height - top
        return image.crop((left, top, right, bottom))

    def extreme_rotate(self, image):
        return image.rotate(random.uniform(-180, 180))

    def extreme_contrast(self, image):
        factor = random.choice([random.uniform(0.1, 0.3), random.uniform(3.0, 5.0)])
        return ImageEnhance.Contrast(image).enhance(factor)

    def extreme_random_pixels(self, image):
        image = self.convert_to_rgb(image)
        num_pixels = random.randint(10000, 100000)
        img_array = np.array(image)
        for _ in range(num_pixels):
            x = np.random.randint(0, img_array.shape[1])
            y = np.random.randint(0, img_array.shape[0])
            img_array[y, x] = np.random.randint(0, 256, size=3)
        return Image.fromarray(img_array)

    def color_inversion(self, image):
        return ImageOps.invert(self.convert_to_rgb(image))

    def extreme_compression(self, image):
        image = self.convert_to_rgb(image)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=random.randint(1, 10))
        buffer.seek(0)
        return Image.open(buffer)

    def random_channels(self, image):
        image = self.convert_to_rgb(image)
        img_array = np.array(image)
        np.random.shuffle(img_array.T)
        return Image.fromarray(img_array)

    def apply_multiple_modifications(self, image, num_modifications=3):
        for _ in range(num_modifications):
            function = random.choice(self.modification_functions[self.mode])
            image = function(image)
        return image


    def process_image_batch(self, batch):
        processed_count = 0
        for input_path, output_path in batch:
            try:
                with Image.open(input_path) as img:
                    modified_img = self.apply_multiple_modifications(img)
                    output_path = os.path.splitext(output_path)[0] + '.jpg'
                    modified_img.save(output_path, "JPEG", quality=85)
                processed_count += 1
            except Exception as e:
                print(f"\nError processing {input_path}: {str(e)}")
        return processed_count

    def get_image_files(self, directory):
        image_files = []
        for file in os.listdir(directory):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_files.append(os.path.join(directory, file))
        return image_files
    
    def process_images_in_folder(self, input_folder, output_folder, num_workers=None):
        print(f"\n{'='*50}")
        print(f"Processing folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        print(f"{'='*50}\n")

        try:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                print(f"Created output folder: {output_folder}")
        except Exception as e:
            print(f"Error creating output folder: {str(e)}")
            return 0

        image_files = self.get_image_files(input_folder)
        total_files = len(image_files)
        
        print(f"Found {total_files} image files")
        if total_files == 0:
            print("No image files found in the input folder.")
            return 0

        if num_workers is None:
            num_workers = cpu_count()
        print(f"Using {num_workers} worker processes")

        batch_size = max(1, min(100, total_files // (num_workers * 4)))
        batches = []
        current_batch = []

        for input_path in image_files:
            output_filename = f"{self.mode}_modified_{os.path.splitext(os.path.basename(input_path))[0]}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            current_batch.append((input_path, output_path))
            
            if len(current_batch) == batch_size:
                batches.append(current_batch)
                current_batch = []

        if current_batch:
            batches.append(current_batch)

        print(f"Created {len(batches)} batches")

        processed_count = 0
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.process_image_batch, batch) for batch in batches]
            
            for future in as_completed(futures):
                processed_count += future.result()
                progress = int((processed_count / total_files) * 100)
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / (processed_count / total_files)
                remaining_time = estimated_total_time - elapsed_time
                print(f"\rProcessed {processed_count}/{total_files} images ({progress}%) - Est. remaining time: {remaining_time:.2f}s", end="", flush=True)

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"\nProcessed {processed_count} out of {total_files} files")
        print(f"Total processing time: {processing_time:.2f} seconds")
        print(f"{'='*50}\n")
        
        return processed_count
