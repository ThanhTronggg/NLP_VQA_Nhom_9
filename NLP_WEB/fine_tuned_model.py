import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import io
import sys
import os
import gc

# Force CPU globally
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_default_device('cpu')
torch.set_default_dtype(torch.float32)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12):
    # Ensure image is in RGB mode
    image = image.convert('RGB')
    
    # Get image dimensions
    w, h = image.size
    
    # If image is too small, don't split
    if w < image_size or h < image_size:
        return [image]
    
    # Calculate number of splits needed
    num_w = w // image_size
    num_h = h // image_size
    
    # Limit total number of splits
    total_splits = num_w * num_h
    if total_splits > max_num:
        scale_factor = (max_num / total_splits) ** 0.5
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        num_w = new_w // image_size
        num_h = new_h // image_size
    
    # Split image into patches
    patches = []
    for i in range(num_h):
        for j in range(num_w):
            left = j * image_size
            top = i * image_size
            right = left + image_size
            bottom = top + image_size
            patch = image.crop((left, top, right, bottom))
            patches.append(patch)
    
    return patches

def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    else:
        image = image_file

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class FineTunedModel:
    def __init__(self, model_path="5CD-AI/Vintern-1B-v2"):
        try:
            # Force CPU mode
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            torch.set_grad_enabled(False)
            
            # Set default device to CPU
            device = torch.device('cpu')
            torch.set_default_device(device)
            
            # Configure model loading for CPU
            model_kwargs = {
                'torch_dtype': torch.float32,
                'low_cpu_mem_usage': True,
                'trust_remote_code': True,
                'device_map': 'cpu',
                'use_safetensors': True
            }
            
            # Load model from Hugging Face
            self.model = AutoModel.from_pretrained(
                model_path,
                **model_kwargs
            ).eval()
            
            # Ensure model is on CPU
            self.model = self.model.to(device)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            # Configure generation parameters
            self.generation_config = {
                'max_new_tokens': 1024,
                'do_sample': False,
                'num_beams': 3,
                'repetition_penalty': 2.5
            }
            
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            print("Loaded Vintern-1B-v2 model successfully in CPU mode")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def process_image(self, image_data):
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Process image
            with torch.no_grad():
                pixel_values = load_image(image, max_num=12)
                pixel_values = pixel_values.to('cpu')
            
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return pixel_values
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise
    
    def get_response(self, question, image_data=None):
        try:
            with torch.no_grad():
                if image_data:
                    pixel_values = self.process_image(image_data)
                    # Ensure inputs are on CPU
                    pixel_values = pixel_values.to('cpu')
                    response, history = self.model.chat(
                        self.tokenizer,
                        pixel_values,
                        question,
                        self.generation_config,
                        history=None,
                        return_history=True
                    )
                else:
                    response, history = self.model.chat(
                        self.tokenizer,
                        None,
                        question,
                        self.generation_config,
                        history=None,
                        return_history=True
                    )
                
                # Clear memory
                gc.collect()
                
                return response
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return f"Lỗi khi xử lý với model Vintern-1B-v2: {str(e)}"
    
    def clear_history(self):
        # InternVLChat model handles history internally
        pass 