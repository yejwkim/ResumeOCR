# brew install poppler
import os
from pdf2image import convert_from_path

# Define the folder containing resumes
pdf_folder = "Resumes"

# Ensure output directory exists
output_folder = "Processed"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        pdf_name = os.path.splitext(filename)[0]
        print(pdf_path)
        print(pdf_name)
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)

        # # Create a subfolder for each PDF's pages
        # pdf_output_folder = os.path.join(output_folder, pdf_name)
        # os.makedirs(pdf_output_folder, exist_ok=True)

        # Save each page as an image
        for i, image in enumerate(images):
            # image_path = os.path.join(pdf_output_folder, f'page_{i + 1}.png')
            image_path = os.path.join(output_folder, f'{pdf_name}-{i + 1}.png')
            image.save(image_path, 'PNG')

        print(f"Converted {filename} -> {len(images)} pages saved in {os.path.join(output_folder, pdf_name)}")
