import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
import time
import concurrent.futures

import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import MTCNN

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("fairface_pipeline.log")],
)
logger = logging.getLogger(__name__)

# --- FairFace Model wrapper ---

class FairFaceModel:
    def __init__(self, model_path: str, output_classes: int, device: str):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path, output_classes)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded model {model_path} on {self.device}")

    def _load_model(self, model_path: str, output_classes: int):
        import torchvision.models as models
        model = models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, output_classes)
        state = torch.load(model_path, map_location=self.device)
        if 'state_dict' in state:
            state = state['state_dict']  # for some checkpoints
        model.load_state_dict(state)
        return model

    def predict_batch(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = torch.stack([self.transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs.cpu()

# --- Face alignment with MTCNN ---

class FaceAligner:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

    def align_faces(self, image_path: str) -> List[Image.Image]:
        """Detect & align faces, return list of cropped face images"""
        try:
            img = Image.open(image_path).convert('RGB')
            boxes, probs = self.mtcnn.detect(img)
            if boxes is None:
                logger.warning(f"No faces detected in {image_path}")
                return []
            faces = []
            for box in boxes:
                # Crop and align face with a margin
                x1, y1, x2, y2 = [int(b) for b in box]
                margin = 20
                w, h = img.size
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                face_img = img.crop((x1, y1, x2, y2))
                faces.append(face_img)
            return faces
        except Exception as e:
            logger.error(f"Face alignment error on {image_path}: {e}")
            return []

# --- FairFace Analyzer combining all 3 models ---

class FairFaceAnalyzer:
    RACE_LABELS = ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian']
    GENDER_LABELS = ['Male', 'Female']
    AGE_LABELS = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

    def __init__(self, race_model_path: str, age_model_path: str, gender_model_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.race_model = FairFaceModel(race_model_path, 7, self.device)
        self.age_model = FairFaceModel(age_model_path, 9, self.device)
        self.gender_model = FairFaceModel(gender_model_path, 2, self.device)

    def analyze_batch(self, images: List[Image.Image]) -> List[Dict]:
        race_probs = self.race_model.predict_batch(images)
        age_probs = self.age_model.predict_batch(images)
        gender_probs = self.gender_model.predict_batch(images)

        results = []
        for i in range(len(images)):
            race_idx = torch.argmax(race_probs[i]).item()
            age_idx = torch.argmax(age_probs[i]).item()
            gender_idx = torch.argmax(gender_probs[i]).item()

            results.append({
                "race": self.RACE_LABELS[race_idx],
                "race_confidence": race_probs[i][race_idx].item(),
                "age": self.AGE_LABELS[age_idx],
                "age_confidence": age_probs[i][age_idx].item(),
                "gender": self.GENDER_LABELS[gender_idx],
                "gender_confidence": gender_probs[i][gender_idx].item(),
            })
        return results

# --- Prompt reconstruction with concurrency ---

class PromptReconstructor:
    def __init__(self, api_key: str, api_url="https://api.openai.com/v1/chat/completions", max_workers=5):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def reconstruct_prompt(self, demographics: Dict, additional_context: str = "") -> str:
        system_prompt = (
            "You are an AI image prompt engineer. Given demographic information extracted from an image, "
            "create a detailed image generation prompt that would likely produce a similar image. "
            "Focus on natural, respectful descriptions."
        )
        user_prompt = (
            f"Based on the following demographic analysis:\n"
            f"- Gender: {demographics.get('gender', 'Unknown')}\n"
            f"- Race/Ethnicity: {demographics.get('race', 'Unknown')}\n"
            f"- Age Range: {demographics.get('age', 'Unknown')}\n\n"
            f"Additional context: {additional_context}\n\n"
            "Create a detailed, respectful image generation prompt that could recreate a similar image. "
            "Focus on artistic style, composition, and natural human features."
        )
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7,
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error generating prompt: {e}"

    def reconstruct_prompts_parallel(self, demographics_list: List[Dict], additional_context: str = "") -> List[str]:
        futures = [
            self.executor.submit(self.reconstruct_prompt, demographics, additional_context)
            for demographics in demographics_list
        ]
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Error in prompt reconstruction thread: {e}")
                results.append("Error generating prompt")
        return results

# --- Utility to gather images ---

def gather_image_paths(paths: List[str]) -> List[str]:
    img_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    all_images = []
    for p in paths:
        path_obj = Path(p)
        if path_obj.is_file() and path_obj.suffix.lower() in img_extensions:
            all_images.append(str(path_obj))
        elif path_obj.is_dir():
            for ext in img_extensions:
                all_images.extend([str(fp) for fp in path_obj.glob(f"*{ext}")])
    return all_images

# --- Main pipeline ---

def main():
    parser = argparse.ArgumentParser(description="FairFace demographic analysis with face alignment and prompt reconstruction")
    parser.add_argument("--images", nargs='+', required=True, help="Image file(s) or directory(s)")
    parser.add_argument("--race-model", required=True, help="Path to FairFace race model")
    parser.add_argument("--age-model", required=True, help="Path to FairFace age model")
    parser.add_argument("--gender-model", required=True, help="Path to FairFace gender model")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--output", default="results.json", help="Output JSON file")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for inference")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for model inference")
    parser.add_argument("--max-workers", type=int, default=5, help="Max concurrent OpenAI API calls")
    args = parser.parse_args()

    images = gather_image_paths(args.images)
    if not images:
        logger.error("No valid images found.")
        sys.exit(1)

    logger.info(f"Processing {len(images)} images on device {args.device}")

    face_aligner = FaceAligner(device=args.device)
    analyzer = FairFaceAnalyzer(args.race_model, args.age_model, args.gender_model, args.device)
    reconstructor = PromptReconstructor(args.api_key, max_workers=args.max_workers)

    all_results = []

    # Process each image file
    for img_path in images:
        logger.info(f"Processing image: {img_path}")
        faces = face_aligner.align_faces(img_path)
        if not faces:
            logger.warning(f"No faces found in image {img_path}, skipping.")
            continue

        # Process in batches if multiple faces in one image
        for i in range(0, len(faces), args.batch_size):
            batch_faces = faces[i:i + args.batch_size]
            demographics_list = analyzer.analyze_batch(batch_faces)
            # Reconstruct prompts in parallel
            prompts = reconstructor.reconstruct_prompts_parallel(demographics_list)

            for face_idx, (demographics, prompt) in enumerate(zip(demographics_list, prompts)):
                result = {
                    "original_image": img_path,
                    "face_index": i + face_idx,
                    "demographics": demographics,
                    "reconstructed_prompt": prompt,
                    "timestamp": time.time(),
                }
                all_results.append(result)
                logger.info(f"Processed face {i + face_idx} in {img_path}")

    # Save all results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Pipeline complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
