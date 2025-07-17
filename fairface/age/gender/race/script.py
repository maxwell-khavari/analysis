import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fairface_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FairFaceAnalyzer:
    """FairFace model wrapper for demographic analysis"""
    
    def __init__(self, race_model_path: str, age_model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load both models
        self.race_model = self._load_model(race_model_path, "race")
        self.age_model = self._load_model(age_model_path, "age") 
        
        self.transform = self._get_transform()
        
        # FairFace categories (based on original FairFace paper)
        self.race_labels = ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian']
        self.gender_labels = ['Male', 'Female']
        self.age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        
        logger.info(f"FairFace models loaded on {self.device}")
    
    def _load_model(self, model_path: str, model_type: str):
        """Load the FairFace model"""
        try:
            # Load the state dict
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Create ResNet34 model architecture
            import torchvision.models as models
            if model_type == "race":
                model = models.resnet34(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, 7)  # 7 race categories
            else:  # age model
                model = models.resnet34(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, 9)  # 9 age categories
            
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            logger.info(f"Loaded {model_type} model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {e}")
            raise
    
    def _get_transform(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def analyze_image(self, image_path: str) -> Dict[str, str]:
        """Analyze a single image and return demographic predictions"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference on both models
            with torch.no_grad():
                # Race prediction
                race_outputs = self.race_model(image_tensor)
                race_pred = torch.nn.functional.softmax(race_outputs, dim=1)
                race_idx = torch.argmax(race_pred, dim=1).item()
                
                # Age prediction (also predicts gender)
                age_outputs = self.age_model(image_tensor)
                age_pred = torch.nn.functional.softmax(age_outputs, dim=1)
                age_idx = torch.argmax(age_pred, dim=1).item()
                
                # Gender prediction (inferred from age model or separate logic)
                # Note: You may need to adjust this based on your specific model outputs
                gender_idx = 0 if age_idx < 4 else 1  # Simplified logic - adjust as needed
                
                return {
                    'race': self.race_labels[race_idx],
                    'gender': self.gender_labels[gender_idx],
                    'age': self.age_labels[age_idx],
                    'race_confidence': race_pred[0][race_idx].item(),
                    'age_confidence': age_pred[0][age_idx].item(),
                    'gender_confidence': 0.7  # Placeholder - adjust based on your model
                }
                
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return {}

class PromptReconstructor:
    """API client for prompt reconstruction"""
    
    def __init__(self, api_key: str, api_url: str = "https://api.openai.com/v1/chat/completions"):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def reconstruct_prompt(self, demographics: Dict[str, str], 
                          additional_context: str = "") -> str:
        """Reconstruct prompt from demographic information"""
        
        # Create prompt for reconstruction
        system_prompt = """You are an AI image prompt engineer. Given demographic information extracted from an image, create a detailed image generation prompt that would likely produce a similar image. Focus on natural, respectful descriptions."""
        
        user_prompt = f"""
        Based on the following demographic analysis:
        - Gender: {demographics.get('gender', 'Unknown')}
        - Race/Ethnicity: {demographics.get('race', 'Unknown')}
        - Age Range: {demographics.get('age', 'Unknown')}
        
        Additional context: {additional_context}
        
        Create a detailed, respectful image generation prompt that could recreate a similar image. Focus on artistic style, composition, and natural human features.
        """
        
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return f"Error: Unable to generate prompt - {str(e)}"
        except KeyError as e:
            logger.error(f"Unexpected API response format: {e}")
            return "Error: Unexpected API response format"

def process_images(image_paths: List[str], 
                  race_model_path: str,
                  age_model_path: str,
                  api_key: str,
                  output_file: str = "reconstructed_prompts.json",
                  device: str = "cuda") -> None:
    """Process multiple images and save results"""
    
    # Initialize components
    analyzer = FairFaceAnalyzer(race_model_path, age_model_path, device)
    reconstructor = PromptReconstructor(api_key)
    
    results = []
    
    for i, image_path in enumerate(image_paths):
        logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        
        # Analyze image
        demographics = analyzer.analyze_image(image_path)
        
        if not demographics:
            logger.warning(f"Skipping {image_path} - analysis failed")
            continue
        
        # Reconstruct prompt
        reconstructed_prompt = reconstructor.reconstruct_prompt(demographics)
        
        # Store results
        result = {
            'image_path': image_path,
            'demographics': demographics,
            'reconstructed_prompt': reconstructed_prompt,
            'timestamp': time.time()
        }
        
        results.append(result)
        
        # Add delay to respect API rate limits
        time.sleep(1)
        
        logger.info(f"Completed: {image_path}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="FairFace Analysis and Prompt Reconstruction")
    parser.add_argument("--images", nargs='+', required=True, 
                       help="Path(s) to image files or directory")
    parser.add_argument("--race-model", required=True, 
                       help="Path to FairFace race model file (res34_fair_align_multi_7_20190809.pt)")
    parser.add_argument("--age-model", required=True, 
                       help="Path to FairFace age model file (res34_fair_align_multi_4_20190809.pt)")
    parser.add_argument("--api-key", required=True, 
                       help="API key for prompt reconstruction")
    parser.add_argument("--output", default="reconstructed_prompts.json", 
                       help="Output file for results")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                       help="Device to use for inference")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Collect image paths
    image_paths = []
    for path in args.images:
        if os.path.isfile(path):
            image_paths.append(path)
        elif os.path.isdir(path):
            # Add all images in directory
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                image_paths.extend(Path(path).glob(ext))
                image_paths.extend(Path(path).glob(ext.upper()))
    
    if not image_paths:
        logger.error("No valid image files found")
        sys.exit(1)
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Process images
    process_images(
        image_paths=image_paths,
        race_model_path=args.race_model,
        age_model_path=args.age_model,
        api_key=args.api_key,
        output_file=args.output,
        device=args.device
    )

if __name__ == "__main__":
    main()


