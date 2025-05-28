import pytesseract
from PIL import Image
import os
import json
from tqdm import tqdm

input_folder = "data/images"
output_folder = "data/ocr_output"
os.makedirs(output_folder, exist_ok=True)
image_base_url = "http://localhost:8080" # Change if necessary
resume_dirs = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
label_studio_tasks = []

for resume_dir in tqdm(resume_dirs, desc="Processing Resumes"):
    resume_path = os.path.join(input_folder, resume_dir)
    image_files = sorted([f for f in os.listdir(resume_path) if f.lower().endswith(".png")])

    for image_file in tqdm(image_files, desc=f"{resume_dir}", leave=False):
        image_path = os.path.join(resume_path, image_file)
        image = Image.open(image_path)
        width, height = image.size
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        results = []
        
        for i, word in enumerate(data["text"]):
            word = word.strip()
            if word:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                x_pct = (x / width) * 100
                y_pct = (y / height) * 100
                w_pct = (w / width) * 100
                h_pct = (h / height) * 100

                results.append({
                    "id": f"{resume_dir}_{i}",
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "x": x_pct,
                        "y": y_pct,
                        "width": w_pct,
                        "height": h_pct,
                        "labels": ["O"]
                    }
                })

        image_url = f"{image_base_url}/{resume_dir}/{image_file}"
        task = {
            "data": {
                "image": image_url
            },
            "annotations": [{
                "result": results
            }]
        }

        label_studio_tasks.append(task)

output_path = os.path.join(output_folder, "import.json")
with open(output_path, "w") as f:
    json.dump(label_studio_tasks, f, indent=2)

print("OCR extraction complete. JSON files saved in 'data/ocr_output/'")