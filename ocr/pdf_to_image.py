# brew install poppler
import os
from pdf2image import convert_from_path
from tqdm import tqdm

pdf_folder = "../data/raw_resumes"
output_folder = "../data/images"
os.makedirs(output_folder, exist_ok=True)
pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

for filename in tqdm(pdf_files, desc="Processing PDFs"):
    pdf_path = os.path.join(pdf_folder, filename)
    pdf_name = os.path.splitext(filename)[0]
    pdf_output_dir = os.path.join(output_folder, pdf_name)
    os.makedirs(pdf_output_dir, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=300) # Convert PDF to images
    
    # Save each page as an image
    for i, image in enumerate(tqdm(images, desc=f"Saving pages of {pdf_name}", leave=False)):
        image_path = os.path.join(pdf_output_dir, f'page-{i + 1}.png')
        image.save(image_path, 'PNG')
