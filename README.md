# IE643 | Team A2A | 22b1520 & 22b1536
----
# Formula Detection and Recognition App

This is a Streamlit-based web application for detecting, segmenting, and recognizing formulas and text from images or PDFs. The app uses advanced OCR techniques, deep learning models, and preprocessing pipelines to classify and extract content, providing LaTeX outputs for formulas and plain text for other content.

---

## Features
- Supports both image and PDF uploads.
- Automatically segments images and extracts text or formulas.
- Classifies content into "Text" or "Formula".
- Converts formulas into LaTeX format for easy use in documents or presentations.
- Handles multi-page PDF files.

---

## Requirements

### Dependencies
Make sure you have the following installed:

- **Python**: >=3.8
- Libraries:
  - `streamlit`
  - `opencv-python`
  - `numpy`
  - `pytesseract`
  - `easyocr`
  - `torch`
  - `transformers`
  - `Pillow`
  - `pdf2image`

To install all dependencies, run:
```bash
pip install -r requirements.txt
```

### Additional Setup
1. **Tesseract**: Install Tesseract OCR for text recognition. [Installation Guide](https://github.com/tesseract-ocr/tesseract)
2. **Poppler**: Required for PDF-to-image conversion. Install via:
   - **Ubuntu**: `sudo apt install poppler-utils`
   - **Mac**: `brew install poppler`
   - **Windows**: Download from [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/).

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/formula-detection-app.git
   cd formula-detection-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up additional resources:
   - Download the pre-trained models used by `VisionEncoderDecoderModel` and `AutoProcessor`:
     ```bash
     from transformers import AutoProcessor, VisionEncoderDecoderModel

     # Replace <model-name> with the appropriate names (ensure internet connectivity)
     VisionEncoderDecoderModel.from_pretrained("<model-name>")
     AutoProcessor.from_pretrained("<model-name>")
     ```

---

## Usage

1. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Access the app in your web browser at `http://localhost:8501`.

### Workflow
- **Upload Files**: Upload an image (`.png`, `.jpg`, `.jpeg`) or a PDF file.
- **Process Files**:
  - For PDFs, each page will be processed individually.
  - The system will:
    - Convert the input to black-and-white.
    - Segment the content into smaller regions.
    - Classify each segment as "Text" or "Formula".
    - Extract and display the recognized text or LaTeX.
- **View Results**:
  - Segmented images are displayed for verification.
  - Recognized content is shown in a scrollable output box.

---

## Hyperparameters

| **Parameter**       | **Description**                                    | **Default Value**      |
|----------------------|----------------------------------------------------|------------------------|
| `FORMULA_THRESHOLD` | Threshold for formula density during segmentation. | `0.7`                 |
| `FORMULA_THRESHOLD_2` | Symbol density to classify segments as formulas. | `0.1`                 |
| `max_segments`      | Maximum number of segments for image processing.   | `10`                  |
| `kernel_size`       | Initial kernel size for morphological operations.  | `30`                  |
| `new_width`/`new_height` | Dimensions for resizing images.                   | `3300x1800`           |

---

## File Structure

├── formula_model/           # Directory containing the formula detection model and related resources
├── .gitignore               # Git configuration to ignore unnecessary files
├── README.md                # Documentation file (this file)
├── app.py                   # Initial app file (currently being refactored, main app is in pdf_reader.py)
├── pdf_reader.py            # Main Streamlit app for formula detection and recognition
├── pipeline.py              # Pipeline script containing core image processing and formula detection functions
├── check_1.png              # Sample image for testing (black and white version)
├── nice.png                 # Sample image for testing (converted to black and white)
├── nice_new.png             # Another sample image for testing (also converted to black and white)
├── requirements.txt         # List of dependencies required for the project
└── .DS_Store                # MacOS system file (should be removed from the repository)

---

## Acknowledgements

- Pre-trained models used from Hugging Face (`AadityaJain/OCR_handwritten`, `hoang-quoc-trung/sumen-base`).
- OCR supported by `Tesseract` and `EasyOCR`.

---

## Troubleshooting

- **Error: Tesseract not found**: Ensure Tesseract is installed and added to your PATH.
- **Error with Poppler**: Install Poppler using the instructions provided above.
- **CUDA not available**: The app will fall back to CPU if GPU is not available.

For further issues, open an issue on the repository.

---

Feel free to update the placeholders (like repository URL or additional resources) as per your project specifics!
