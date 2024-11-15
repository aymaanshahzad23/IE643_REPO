# Formula Detection and Recognition App

A Streamlit-based application for detecting, segmenting, and recognizing mathematical formulas and text from images and PDFs. The app leverages OCR techniques and deep learning models to classify content and generate LaTeX output for formulas.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Configuration Parameters](#configuration-parameters)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- Python >= 3.8
- GPU (optional, will fall back to CPU if unavailable)

### Required Software
1. **Tesseract OCR**
   - Ubuntu: `sudo apt-get install tesseract-ocr`
   - MacOS: `brew install tesseract`
   - Windows: Download installer from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

2. **Poppler** (for PDF processing)
   - Ubuntu: `sudo apt install poppler-utils`
   - MacOS: `brew install poppler`
   - Windows: Download from [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/formula-detection-app.git
   cd formula-detection-app
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models**
   ```python
   from transformers import AutoProcessor, VisionEncoderDecoderModel
   
   # Download OCR model
   VisionEncoderDecoderModel.from_pretrained("AadityaJain/OCR_handwritten")
   AutoProcessor.from_pretrained("AadityaJain/OCR_handwritten")
   
   # Download formula recognition model
   VisionEncoderDecoderModel.from_pretrained("hoang-quoc-trung/sumen-base")
   AutoProcessor.from_pretrained("hoang-quoc-trung/sumen-base")
   ```

## Running the Application

1. **Launch the Streamlit App**
   ```bash
   streamlit run pdf_reader.py
   ```

2. **Access the Interface**
   - Open your web browser
   - Navigate to `http://localhost:8501`

3. **Using the Application**
   1. Upload your input file:
      - Supported image formats: `.png`, `.jpg`, `.jpeg`
      - Supported document format: `.pdf`
   2. Wait for processing to complete
   3. View results in the interface:
      - Segmented images
      - Recognized text
      - LaTeX formulas

## Data Processing Pipeline

The application processes inputs through the following stages:

1. **Image Preprocessing**
   - Conversion to black and white
   - Resizing to standard dimensions
   - Noise reduction

2. **Content Segmentation**
   - Region detection
   - Content classification (Text/Formula)

3. **Recognition**
   - Text extraction using Tesseract OCR
   - Formula conversion to LaTeX

## Configuration Parameters

### Critical Parameters
| Parameter | Description | Default Value | Recommended Range |
|-----------|-------------|---------------|------------------|
| `FORMULA_THRESHOLD` | Formula density threshold for segmentation | 0.7 | 0.6-0.8 |
| `FORMULA_THRESHOLD_2` | Symbol density for formula classification | 0.1 | 0.05-0.15 |
| `max_segments` | Maximum segments per image | 10 | 5-15 |
| `kernel_size` | Morphological operation kernel size | 30 | 20-40 |
| `new_width` | Image resize width | 3300 | 2000-4000 |
| `new_height` | Image resize height | 1800 | 1000-2000 |

### Adjusting Parameters
- Edit these values in `pipeline.py` before running the application
- Higher `FORMULA_THRESHOLD` values increase precision but may miss some formulas
- Lower `max_segments` values improve processing speed but may reduce accuracy

## Project Structure
```
formula-detection-app/
├── formula_model/           # Formula detection models
├── app.py                   # Legacy app file
├── pdf_reader.py            # Main Streamlit application
├── pipeline.py             # Core processing functions
├── requirements.txt        # Project dependencies
└── README.md              # Documentation
```

## Troubleshooting

### Common Issues and Solutions

1. **Tesseract Not Found Error**
   ```
   Solution: Add Tesseract to your system PATH:
   - Windows: Add C:\Program Files\Tesseract-OCR
   - Linux/Mac: Verify installation with 'tesseract --version'
   ```

2. **PDF Processing Error**
   ```
   Solution: Verify Poppler installation:
   - Windows: Add poppler/bin to PATH
   - Linux/Mac: Run 'pdftocairo -v'
   ```

3. **CUDA/GPU Issues**
   ```
   Solution: The app automatically uses CPU if CUDA is unavailable.
   To use GPU:
   - Install CUDA Toolkit
   - Install appropriate PyTorch version
   ```

### Getting Help
- Open an issue on the repository
- Include error messages and system information
- Provide sample input files when possible

For additional assistance or feature requests, please create an issue in the repository.
