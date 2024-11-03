import cv2
import numpy as np
import pytesseract
import easyocr

# Initialize EasyOCR reader/
reader = easyocr.Reader(['en'])

# Define symbols typical of math formulas
math_symbols = set([
    # Greek letters (lowercase)
    'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
    # Greek letters (uppercase)
    'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω',
    # Arithmetic operators
    '+', '-', '*', '/', '=', '≠', '≈', '<', '>', '≤', '≥', '±', '∓', '∙', '×', '÷', '∝',
    # Logic and set symbols
    '∧', '∨', '∩', '∪', '∈', '∉', '⊂', '⊃', '⊆', '⊇', '∅', '∃', '∀',
    # Calculus symbols
    '∂', '∇', '∫', '∬', '∮', '∑', '∏', '∆', '∫', '∬', '∮', '∑', '∏', '∂',
    # Trigonometric functions (if extracted as words)
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    # Special symbols
    '°', '′', '″', '|', '‖', '∥', '|||', '∠', '∢', '⊥', '∥',
    # Miscellaneous
    '∞', '√', '∛', '∜', '∴', '∵', '∓', '∠', '∡', '∢',
    # Set notation and imaginary symbols
    'ℝ', 'ℚ', 'ℤ', 'ℕ', 'ℂ', 'ℍ', '℘',
    # Probability and statistical symbols
    'E', 'Var', 'Cov', 'P', 'Pr',
    # Advanced inequalities and relations
    '≪', '≫', '∼', '≅', '≈',
    # Binary and set operations
    '⊗', '⊕', '⊖', '⊘', '⊙'
])

IMAGE_INPUT="check_1.png"
SEGMENT_IMAGE_SET=[]
FORMULA_THRESHOLD = 0.7
FORMULA_THRESHOLD_2 = 0.05

def iterative_segment_image(image, max_segments=10, density_threshold=FORMULA_THRESHOLD):
    """
    Iteratively segments the image until either the number of segments meets or exceeds max_segments
    or the formula density drops below a specified density_threshold.
    """
    segment_count = 0
    kernel_size = 30  # Starting kernel size
    iteration = 1
    formula_density = 1  # Start with a high formula density

    while segment_count < max_segments and kernel_size > 25 and formula_density > density_threshold:
        # Step 1: Pre-process the image with adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Step 2: Morphological operations with adjustable kernel size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Step 3: Perform connected-component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

        # Ignore background component
        segment_count = num_labels - 1
        print(f"Iteration {iteration}: Kernel Size {kernel_size}, Segments {segment_count}")

        # Estimate formula density for the current segmentation
        formula_density = estimate_formula_density(closed, stats)
        print(f"Formula Density: {formula_density}")

        # Adjust kernel size for finer segmentation in the next iteration based on formula density
        kernel_size_adjustment = -5 if formula_density > FORMULA_THRESHOLD else -2
        kernel_size += kernel_size_adjustment
        iteration += 1


    # Draw bounding boxes on the final segmented result
    image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 500:  # Filter out small areas if necessary
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save the segment if needed
            segmented_part = image[y:y + h, x:x + w]
            segment_filename = f'segment_{i}.png'
            cv2.imwrite(segment_filename, segmented_part)
            print(f'Saved: {segment_filename}')
            SEGMENT_IMAGE_SET.append(segment_filename)

    return image_with_boxes

def estimate_formula_density(image, stats):
    """
    Estimates the formula density in the given segmented image regions.
    """
    formula_symbols_count = 0
    total_chars = 0

    for i in range(1, len(stats)):
        x, y, w, h, area = stats[i]
        region = image[y:y + h, x:x + w]

        # Extract text using Tesseract OCR for simplicity
        reader = easyocr.Reader(['en'])  # 'en' is for English; add more languages as needed

        # Run OCR on the image
        results = reader.readtext(region, detail=0)  # `detail=0` returns text only, not bounding boxes
        text = " ".join(results)  # Join the text results if needed

        # Count formula symbols
        formula_symbols_count += sum(1 for char in text if char in math_symbols)
        total_chars += len(text)

    # Avoid division by zero
    if total_chars == 0:
        return 0

    return formula_symbols_count / total_chars

def preprocess_image(image_path):
    """
    Preprocess the image to improve OCR accuracy.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh_image

def extract_text(image, use_easyocr=False):
    """
    Extract text from the image using either Tesseract or EasyOCR.
    """
    if use_easyocr:
        results = reader.readtext(image)
        extracted_text = ' '.join([res[1] for res in results])
    else:
        extracted_text = pytesseract.image_to_string(image)
    return extracted_text

def classify_text_type(extracted_text):
    """
    Classify the text type as 'text' or 'formula' based on symbol density.
    """
    symbol_count = sum(1 for char in extracted_text if char in math_symbols)
    total_chars = len(extracted_text)
    # Avoid division by zero
    if total_chars == 0:
        return "Unknown"
    # Calculate the ratio of math symbols
    symbol_density = symbol_count / total_chars
    return "Formula" if symbol_density >= FORMULA_THRESHOLD_2 else "Text"

def classify_image(image_path, use_easyocr=False):
    """
    Full pipeline to classify if the image contains regular text or a mathematical formula.
    """
    # Step 1: Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    # Step 2: Extract text from the preprocessed image
    extracted_text = extract_text(preprocessed_image, use_easyocr=use_easyocr)
    # Step 3: Classify based on symbol density
    classification = classify_text_type(extracted_text)
    return classification

from PIL import Image

def resize_image(input_path, output_path, new_width, new_height):
    # Open an image file
    with Image.open(input_path) as img:
        # Resize image
        resized_img = img.resize((new_width, new_height))
        # Save it to a new file
        resized_img.save(output_path)
        print(f"Image saved to {output_path} with size {new_width}x{new_height}")

# Example usage
input_path = IMAGE_INPUT #IMAGE INPUT
output_path = 'NEXT_PIL.png'
new_width = 1300
new_height = 900

resize_image(input_path, output_path, new_width, new_height)

# Load your image here (make sure the path is correct)
image_path = 'NEXT_PIL.png'  # Replace with the path to your image file
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale

# Step 1: Segment the image iteratively
segmented_image = iterative_segment_image(image, max_segments=10)

# Optiona/l: Display the segmented image
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 2: Save the segmented image with bounding boxes

output_path = 'segmented_with_boxes.png'  # Output path for the segmented image
cv2.imwrite(output_path, segmented_image)
print(f'Segmented image saved to: {output_path}')

import torch
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

# Load model & processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained('/content/formula_model', 
    local_files_only=True, 
    use_safetensors=True
).to(device)
processor = AutoProcessor.from_pretrained('hoang-quoc-trung/sumen-base')

def generate_latex_from_image(image_path: str) -> str:
    """
    Generates LaTeX expression from an image of a formula.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: LaTeX expression generated from the image.
    """
    # Task prompt and decoder input IDs
    task_prompt = processor.tokenizer.bos_token
    decoder_input_ids = processor.tokenizer(
        task_prompt,
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids

    # Load and process the image
    image = Image.open(image_path).convert("RGB")  # Convert image to RGB format
    pixel_values = processor.image_processor(
        image,
        return_tensors="pt",
        data_format="channels_first",
    ).pixel_values

    # Generate LaTeX expression
    with torch.no_grad():
        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=4,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    # Decode and clean up the output sequence
    sequence = processor.tokenizer.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(
        processor.tokenizer.eos_token, ""
    ).replace(
        processor.tokenizer.pad_token, ""
    ).replace(processor.tokenizer.bos_token, "")

    return sequence




# Initialize the EasyOCR reader for English
reader = easyocr.Reader(['en'])

def extract_text_from_image(image_path: str) -> str:
    """
    Extracts text from an image using EasyOCR.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Extracted text from the image.
    """
    # Perform OCR on the image
    results = reader.readtext(image_path)

    # Extract the text and join it into a single string
    extracted_text = ' '.join([text for (_, text, _) in results])
    return extracted_text

L = {}
for i in SEGMENT_IMAGE_SET:
  classification = classify_image(i, use_easyocr=True)
  print(f'Classification{i}: {classification}')
  L[i]=classification
print(L)

for i in L:
  if L[i] == "Formula":
    latex_expression = generate_latex_from_image(i)
    print(latex_expression)
  else:
    text = extract_text_from_image(i)
    print(text)


# # Perform OCR on the image
#     results = reader.readtext(image_path)

# # Extract the text and join it into a single string
#     extracted_text = ' '.join([text for (_, text, _) in results])
#     print(extracted_text)