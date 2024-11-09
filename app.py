import streamlit as st
import cv2
import numpy as np
import pytesseract
import easyocr
import torch
from PIL import Image
import os
import tempfile
from transformers import AutoProcessor, VisionEncoderDecoderModel

# Set page config
st.set_page_config(
    page_title="Formula Detection App",
    layout="wide"
)

# Initialize session state for temporary files
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []


def convert_to_black_and_white(image_path):
    image = Image.open(image_path)
    bw_image = image.convert('L')
    threshold = 130  # You can adjust this value
    binary_image = bw_image.point(lambda p: 255 if p > threshold else 0)
    
    # Save the binary image to a temporary file
    binary_image_path = create_temp_file(suffix=".png")
    binary_image.save(binary_image_path)
    return binary_image_path

def cleanup_temp_files():
    """Clean up all temporary files created during the process."""
    if 'temp_files' in st.session_state:
        for temp_file in st.session_state.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"Deleted temporary file: {temp_file}")
                else:
                    print(f"Temporary file not found: {temp_file}")
            except Exception as e:
                print(f"Error deleting file {temp_file}: {e}")
        st.session_state.temp_files = []  # Clear the list after cleanup
    else:
        print("No temporary files to clean up.")


def create_temp_file(suffix=".png"):
    """Create a temporary file and track it for cleanup"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    st.session_state.temp_files.append(temp_file.name)
    return temp_file.name

# Your original code and functions unchanged
SEGMENT_IMAGE_SET = []
FORMULA_THRESHOLD = 0.7
FORMULA_THRESHOLD_2 = 0.05

# Copy of your original math_symbols set
math_symbols = set([
    'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
    'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω',
    '+', '-', '*', '/', '=', '≠', '≈', '<', '>', '≤', '≥', '±', '∓', '∙', '×', '÷', '∝',
    '∧', '∨', '∩', '∪', '∈', '∉', '⊂', '⊃', '⊆', '⊇', '∅', '∃', '∀',
    '∂', '∇', '∫', '∬', '∮', '∑', '∏', '∆', '∫', '∬', '∮', '∑', '∏', '∂',
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    '°', '′', '″', '|', '‖', '∥', '|||', '∠', '∢', '⊥', '∥',
    '∞', '√', '∛', '∜', '∴', '∵', '∓', '∠', '∡', '∢',
    'ℝ', 'ℚ', 'ℤ', 'ℕ', 'ℂ', 'ℍ', '℘',
    'E', 'Var', 'Cov', 'P', 'Pr',
    '≪', '≫', '∼', '≅', '≈',
    '⊗', '⊕', '⊖', '⊘', '⊙'
])

# Your original functions exactly as they were
def iterative_segment_image(image, max_segments=10, density_threshold=FORMULA_THRESHOLD):
    """Your original iterative_segment_image function"""
    segment_count = 0
    kernel_size = 30
    iteration = 1
    formula_density = 1

    while segment_count < max_segments and kernel_size > 26 and formula_density > density_threshold:
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (300, kernel_size))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

        segment_count = num_labels - 1
        # print(f"Iteration {iteration}: Kernel Size {kernel_size}, Segments {segment_count}")

        formula_density = estimate_formula_density(closed, stats)
        # print(f"Formula Density: {formula_density}")

        kernel_size_adjustment = -5 if formula_density > FORMULA_THRESHOLD else -2
        kernel_size += kernel_size_adjustment
        iteration += 1

    image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 500:
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

            segmented_part = image[y:y + h, x:x + w]
            segment_filename = create_temp_file(f"_segment_{i}.png")
            cv2.imwrite(segment_filename, segmented_part)
            print(f'Saved: {segment_filename}')
            SEGMENT_IMAGE_SET.append(segment_filename)

    return image_with_boxes

def estimate_formula_density(image, stats):
    """Your original estimate_formula_density function"""
    formula_symbols_count = 0
    total_chars = 0

    for i in range(1, len(stats)):
        x, y, w, h, area = stats[i]
        region = image[y:y + h, x:x + w]

        reader = easyocr.Reader(['en'])
        results = reader.readtext(region, detail=0)
        text = " ".join(results)

        formula_symbols_count += sum(1 for char in text if char in math_symbols)
        total_chars += len(text)

    return formula_symbols_count / max(total_chars, 1)

def preprocess_image(image_path):
    """Preprocess the image and save it as a temporary file"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Create a temporary file to save the thresholded image
    preprocessed_image_path = create_temp_file(suffix=".png")
    cv2.imwrite(preprocessed_image_path, thresh_image)  # Save thresholded image
    return preprocessed_image_path  # Return the path of the processed image

def extract_text(image, use_easyocr=False):
    """Your original extract_text function"""
    reader = easyocr.Reader(['en'])
    if use_easyocr:
        results = reader.readtext(image)
        extracted_text = ' '.join([res[1] for res in results])
    else:
        extracted_text = pytesseract.image_to_string(image)
    return extracted_text

def classify_text_type(extracted_text):
    """Your original classify_text_type function"""
    symbol_count = sum(1 for char in extracted_text if char in math_symbols)
    total_chars = len(extracted_text)
    if total_chars == 0:
        return "Unknown"
    symbol_density = symbol_count / total_chars
    return "Formula" if symbol_density >= FORMULA_THRESHOLD_2 else "Text"

def classify_image(image_path, use_easyocr=False):
    """Classify the content of a binary-preprocessed image as Formula or Text"""
    preprocessed_image_path = preprocess_image(image_path)  # This now returns a path
    preprocessed_image = cv2.imread(preprocessed_image_path, cv2.IMREAD_GRAYSCALE)
    extracted_text = extract_text(preprocessed_image, use_easyocr=use_easyocr)
    classification = classify_text_type(extracted_text)
    return classification


def resize_image(input_path, output_path, new_width, new_height):
    """Your original resize_image function"""
    with Image.open(input_path) as img:
        resized_img = img.resize((new_width, new_height))
        resized_img.save(output_path)
        print(f"Image saved to {output_path} with size {new_width}x{new_height}")

@st.cache_resource
def load_models():
    """Load and cache the models with your original configuration"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionEncoderDecoderModel.from_pretrained('AadityaJain/OCR_handwritten', 

        use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained('hoang-quoc-trung/sumen-base')
    return model, processor

def generate_latex_from_image(image_path: str, model, processor) -> str:
    """Your original generate_latex_from_image function"""
    task_prompt = processor.tokenizer.bos_token
    decoder_input_ids = processor.tokenizer(
        task_prompt,
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor.image_processor(
        image,
        return_tensors="pt",
        data_format="channels_first",
    ).pixel_values

    device = next(model.parameters()).device
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

    sequence = processor.tokenizer.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(
        processor.tokenizer.eos_token, ""
    ).replace(
        processor.tokenizer.pad_token, ""
    ).replace(
        processor.tokenizer.bos_token, ""
    )

    return sequence

def extract_text_from_image(image_path: str) -> str:
    """Your original extract_text_from_image function"""
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    extracted_text = ' '.join([text for (_, text, _) in results])
    return extracted_text

def main():
    st.title("Formula Detection and Recognition App")

    # Load models
    model, processor = load_models()

    # File upload
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "pdf"])

    if uploaded_file is not None:
        SEGMENT_IMAGE_SET.clear()  # Clear segment list at the start

        # Check if the uploaded file is a PDF
        if uploaded_file.type == "application/pdf":
            # Create a temporary file for the PDF
            pdf_temp_file = create_temp_file(suffix=".pdf")
            with open(pdf_temp_file, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Convert PDF pages to images
            pdf_image_files = convert_pdf_to_images(pdf_temp_file)
            SEGMENT_IMAGE_SET.extend(pdf_image_files)  # Add the converted images to the segment list

            # Process each image in the segment list
            for image_path in SEGMENT_IMAGE_SET:
                # Convert the uploaded image to black and white
                bw_image_path = convert_to_black_and_white(image_path)

                # Display original image
                st.subheader("Original Image")
                st.image(image_path)

                # Process image
                if st.button(f"Process Image: {image_path}"):
                    with st.spinner("Processing image..."):
                        # Resize image
                        output_path = create_temp_file()
                        resize_image(bw_image_path, output_path, 1200, 2000)

                        # Load and process image
                        image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

                        # Segment image using your original function
                        segmented_image = iterative_segment_image(image)

                        # Display segmented image
                        st.subheader("Segmented Image")
                        st.image(segmented_image)

                        # Process segments using your original classification
                        L = {}
                        for img_path in SEGMENT_IMAGE_SET:
                            classification = classify_image(img_path, use_easyocr=True)
                            print(f'Classification {img_path}: {classification}')
                            L[img_path] = classification

                        # Generate results as in your original code
                        result = []
                        for img_path in L:
                            if L[img_path] == "Formula":
                                latex_expression = generate_latex_from_image(img_path, model, processor)
                                formulas = latex_expression.split("\\newline")
                                for formula in formulas:
                                    result.append(f"\\begin{{equation}}\n{formula}\n\\end{{equation}}")
                            else:
                                text = extract_text_from_image(img_path)
                                result.append(f"\n{text}\n")

                        # Display final output as in your original code
                        final_output = "\n%----\n".join(result)
                        final_output.replace("%", "\%")
                        st.subheader("Complete Output")
                        col1, col2 = st.columns([4, 1])

                        # Display the text area in the first column
                        with col1:
                            st.text_area("", value=final_output, height=500, key="output_area")
            # Cleanup after processing all images
            cleanup_temp_files()

        else:
            # Create temporary file for input image
            input_path = create_temp_file()
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Convert the uploaded image to black and white
            bw_image_path = convert_to_black_and_white(input_path)

            # Display original image
            st.subheader("Original Image")
            st.image(input_path)

            # Process image
            if st.button("Process Image"):
                with st.spinner("Processing image..."):
                    # Resize image
                    output_path = create_temp_file()
                    resize_image(bw_image_path, output_path, 900, 2000)

                    # Load and process image
                    image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

                    # Segment image using your original function
                    segmented_image = iterative_segment_image(image)

                    # Display segmented image
                    st.subheader("Segmented Image")
                    st.image(segmented_image)

                    # Process segments using your original classification
                    L = {}
                    for img_path in SEGMENT_IMAGE_SET:
                        classification = classify_image(img_path, use_easyocr=True)
                        print(f'Classification {img_path}: {classification}')
                        L[img_path] = classification

                    # Generate results as in your original code
                    result = []
                    for img_path in L:
                        if L[img_path] == "Formula":
                            latex_expression = generate_latex_from_image(img_path, model, processor)
                            formulas = latex_expression.split("\\newline")
                            for formula in formulas:
                                result.append(f"\\begin{{equation}}\n{formula}\n\\end{{equation}}")
                        else:
                            text = extract_text_from_image(img_path)
                            result.append(f"\n{text}\n")

                    # Display final output as in your original code
                    final_output = "\n".join(result)

                    st.subheader("Complete Output")
                    col1, col2 = st.columns([4, 1])

                    # Display the text area in the first column
                    with col1:
                        st.markdown(f"```\n{final_output}\n```", unsafe_allow_html=True)
            # Cleanup after processing the single image
            cleanup_temp_files()


if __name__ == "__main__":
    main()
