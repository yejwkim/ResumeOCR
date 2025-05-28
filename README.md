# ResumeOCR

A preprocessing pipeline for structured resume annotation using OCR and Label Studio. The system converts resumes (PDFs) into labeled image data to prepare for training LayoutLMv3 on entity extraction.

## 📁 Project Structure (To Be Updated)
```bash
resume-ocr/
├── data
│   ├── raw_resumes         # Input: PDF resumes
│   ├── images              # Output: converted images (by page, per resume)
│   └── ocr_output
│       └── import.json     # Output: import.json for Label Studio
├── model
├── ocr
│   ├── cors_http_server.py # Local image hosting (CORS enabled)
│   ├── pdf_to_image.py     # PDF to image conversion
│   ├── run_ocr.py          # OCR text + bounding box extraction
│   └── sample.xml          # BIO tagging template for Label Studio
└── requirements.txt
```

## ✅ Prerequisities
Create a virtual environment (Recommended):
```bash
python -m venv venv
source venv/bin/activate
```

Install required packages:
```bash
pip install -r requirements.txt
```

## 🧾 OCR Extraction Workflow
### 1. Prepare PDF Resumes
Place all resume PDFs in the `data/raw_resumes/` directory.

### 2. Convert PDFs to Images
Run:
```bash
python ocr/pdf_to_image.py
```
Note:
- Each PDF will be converted to one or more PNG images.
- Images will be stored in `data/images/<resume_name>/page-<n>.png`.

### 3. Extract OCR Annotations
Run:
```bash
python ocr/run_ocr.py
```
This will:
- Run Tesseract on each image.
- Extract bounding boxes and text.
- Generate an `import.json` file in `data/ocr_output/` for Label Studio.

## 🏷️ Labeling with Label Studio
### 1. Serve Images Locally
In a terminal:
```bash
cd data/images
python ../../ocr/cors_http_server.py
```
This exposes images on `http://localhost:8080`.

### 2. Launch Label Studio
In another terminal:
```bash
label-studio start
```
Go to `http://localhost:8081`, log in, and create a new project.

### 3. Configure the Project
- **Import**: Upload the `import.json` file from `data/ocr_output/`.
- **Labeling Setup**: Select *"Optical Character Recognition"* template.
- **Tag Schema**: Use the XML from `ocr/sample.xml` for BIO tag definitions. Edit as needed.

    > **Note:** All bounding boxes are pre-tagged as `"O"` (Outside). Update each to its correct BIO tag during labeling.

### 4. Begin Labeling
- Click on bounding boxes to assign proper BIO tags (e.g., `B-Skill`, `I-Education`, etc.).
- Save annotations after completing each page.

## 📤 Exporting Labeled Data
🚧 [Coming Soon]
- Export annotations in JSON format
- Convert to Hugging Face `datasets.Dataset` format for LayoutLMv3 training

## 🤖 Model Training: LayoutLMv3
🚧 [Coming Soon]
- Data preprocessing for `LayoutLMv3Processor`
- Token classification model training
- Evaluation metrics and SHAP-based explainability

## 📌 Notes
- This system supports multi-page resumes for flexibility, though single-page resumes are typically recommended.
- BIO tagging ensures compatibility with transformer-based token classification models like LayoutLMv3.