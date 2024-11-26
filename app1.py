import streamlit as st
import cv2
import numpy as np
import pytesseract
import torch
from PIL import Image
import os
import tempfile
from transformers import AutoProcessor, VisionEncoderDecoderModel, TrOCRProcessor
import streamlit as st
from pdf2image import convert_from_path

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
FORMULA_THRESHOLD_2 = 0.11


math_symbols = set([
    '\\alpha', '\\beta', '\\gamma', '\\delta', '\\epsilon', '\\zeta', '\\eta', '\\theta', '\\iota', '\\kappa',
    '\\lambda', '\\mu', '\\nu', '\\xi', '\\omicron', '\\pi', '\\rho', '\\sigma', '\\tau', '\\upsilon', '\\phi', '\\chi',
    '\\psi', '\\omega',
    '\\Alpha', '\\Beta', '\\Gamma', '\\Delta', '\\Epsilon', '\\Zeta', '\\Eta', '\\Theta', '\\Iota', '\\Kappa',
    '\\Lambda', '\\Mu', '\\Nu', '\\Xi', '\\Omicron', '\\Pi', '\\Rho', '\\Sigma', '\\Tau', '\\Upsilon', '\\Phi',
    '\\Chi', '\\Psi', '\\Omega',
    '+', '-', '*', '/', '=', '\\neq', '\\approx', '<', '>', '\\leq', '\\geq', '\\pm', '\\mp', '\\cdot', '\\times',
    '\\div', '\\propto',
    '\\land', '\\lor', '\\cap', '\\cup', '\\in', '\\notin', '\\subset', '\\supset', '\\subseteq', '\\supseteq',
    '\\emptyset', '\\exists', '\\forall',
    '\\partial', '\\nabla', '\\int', '\\iint', '\\oint', '\\sum', '\\prod', '\\Delta', '\\int', '\\iint', '\\oint',
    '\\sum', '\\prod', '\\partial',
    '\\sin', '\\cos', '\\tan', '\\cot', '\\sec', '\\csc',
    '^\\circ', '\\\'', '\\"', '|', '\\|', '\\parallel', '\\angle', '\\measuredangle', '\\perp',
    '\\infty', '\\sqrt{}', '\\sqrt[3]{}', '\\sqrt[4]{}', '\\therefore', '\\because', '\\mp', '\\sphericalangle',
    '\\mathbb{R}', '\\mathbb{Q}', '\\mathbb{Z}', '\\mathbb{N}', '\\mathbb{C}', '\\mathbb{H}', '\\mathfrak{P}',
    'E', '\\text{Var}', '\\text{Cov}', '\\text{P}', '\\text{Pr}',
    '\\ll', '\\gg', '\\sim', '\\cong', '\\approx',
    '\\otimes', '\\oplus', '\\ominus', '\\oslash', '\\odot', "\\neq", "\\leq", "\geq",
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'x', 'y', 'd'
])

math_weights = {
    # High weights for formula-defining symbols
    '+': 5, '-': 5, '*': 5, '/': 5, '=': 5, '<': 5, '>': 5, '\\neq': 5, '\\leq': 5, '\\geq': 5, '\\approx': 5, '\\sim': 5, 
    '\\equiv': 5, '\\propto': 5, '\\pm': 5, '\\mp': 5, '\\cdot': 5, '\\times': 5, '\\div': 5,
    
    # Medium weights for common mathematical terms and operations
    '\\int': 2, '\\iint': 4, '\\oint': 4, '\\sum': 4, '\\prod': 4, '\\partial': 4, '\\nabla': 4, '\\sqrt': 3, 
    '\\log': 3, '\\ln': 4, '\\exp': 4, '\\sin': 3, '\\cos': 3, '\\tan': 3, '\\cot': 3, '\\sec': 3, '\\csc': 3, 
    '\\arcsin': 3, '\\arccos': 3, '\\arctan': 3,
    
    # Weights for Greek letters
    '\\alpha': 3, '\\beta': 3, '\\gamma': 3, '\\delta': 3, '\\epsilon': 3, '\\zeta': 3, '\\eta': 3, '\\theta': 3,
    '\\kappa': 3, '\\lambda': 3, '\\mu': 3, '\\nu': 3, '\\xi': 3, '\\pi': 3, '\\rho': 3, '\\sigma': 3, '\\tau': 3, 
    '\\phi': 3, '\\chi': 3, '\\psi': 3, '\\omega': 3, '\\Gamma': 3, '\\Delta': 3, '\\Theta': 3, '\\Lambda': 3, 
    '\\Xi': 3, '\\Pi': 3, '\\Sigma': 3, '\\Phi': 3, '\\Psi': 3, '\\Omega': 3,
    
    # Weights for text-like variables (lower, since they can appear in both text and formulas)
    'i': 2, 'j': 2, 'x': 2, 'y': 2, 'z': 2, 'a': 2, 'b': 2, 'c': 2, 'p': 2, 'q': 2,
    
    # Weights for miscellaneous math symbols
    '\\ldots': 3, '\\cdots': 3, '\\vdots': 3, '\\ddots': 3, '\\infty': 3, '\\binom': 3,
    '\\langle': 3, '\\rangle': 3, '\\lceil': 3, '\\rceil': 3, '\\lfloor': 3, '\\rfloor': 3,
    
    # Default for numbers
    '0': 2, '1': 2, '2': 2, '3': 2, '4': 2, '5': 2, '6': 2, '7': 2, '8': 2, '9': 2,
    
    # Delimiters (low, as they are ambiguous)
    '(': 1, ')': 1, '[': 1, ']': 1, '{': 1, '}': 1, '|': 1,
    
    # Text and formatting (very low weights)
    '\\text': 1, '\\mathrm': 1, '\\mathbf': 1, '\\mathit': 1, '\\mathsf': 1, '\\mathtt': 1,
    
    # Special mathematical constants
    '\\mathbb{R}': 3, '\\mathbb{Z}': 3, '\\mathbb{N}': 3, '\\mathbb{Q}': 3, '\\mathbb{C}': 3
}

# math_symbols = set([
#     # Greek Letters (Lowercase and Uppercase)
#     '\\alpha', '\\beta', '\\gamma', '\\delta', '\\epsilon', '\\zeta', '\\eta', '\\theta', '\\iota', '\\kappa',
#     '\\lambda', '\\mu', '\\nu', '\\xi', '\\omicron', '\\pi', '\\rho', '\\sigma', '\\tau', '\\upsilon', '\\phi', '\\chi',
#     '\\psi', '\\omega', '\\varepsilon', '\\vartheta', '\\varphi', '\\varsigma', '\\varpi',
#     '\\Gamma', '\\Delta', '\\Theta', '\\Lambda', '\\Xi', '\\Pi', '\\Sigma', '\\Upsilon', '\\Phi', '\\Psi', '\\Omega',
    
#     # Basic Arithmetic
#     '+', '-', '*', '/', '=', '<', '>', '\\neq', '\\sim', '\\equiv', '\\leq', '\\geq',
    
#     # # Logic and Set Theory
#     # '\\lor', '\\neg', '\\iff', '\\forall', '\\exists', '\\nexists', '\\in', '\\notin',
#     # '\\ni', '\\subset', '\\supset', '\\subseteq', '\\supseteq', '\\cap', '\\cup', '\\emptyset', '\\setminus',
#     # '\\complement', '\\overline',
    
#     # Calculus and Limits
#     '\\partial', '\\nabla', '\\int', '\\iint', '\\iiint', '\\oint', '\\sum', '\\prod', '\\lim', '\\infty', '\\frac',
#     '\\binom', '\\sqrt', '\\sqrt[3]', '\\sqrt[4]',
    
#     # Trigonometry and Functions
#     '\\sin', '\\cos', '\\tan', '\\cot', '\\sec', '\\csc', '\\arcsin', '\\arccos', '\\arctan',
#     '\\log', '\\ln', '\\exp',
    
#     # # Arrows and Relations
#     # '\\to', '\\rightarrow', '\\leftarrow', '\\leftrightarrow', '\\uparrow', '\\downarrow', '\\Rightarrow',
#     # '\\Leftarrow', '\\Leftrightarrow', '\\mapsto', '\\hookrightarrow', '\\longrightarrow', '\\longleftarrow',
#     # '\\longleftrightarrow',
    
#     # # Miscellaneous Symbols
#     # '\\circ', '\\diamond', '\\star', '\\ast', '\\bigstar', '\\bullet', '\\prime', '\\dagger', '\\ddagger',
#     # '\\angle', '\\measuredangle', '\\sphericalangle', '\\parallel', '\\perp', '|', '\\|', '\\backslash',
#     '\\ldots', '\\cdots', '\\vdots', '\\ddots', '\\aleph',
    
#     # Over and Under Accents
#     '\\hat', '\\widehat', '\\bar', '\\overline', '\\underline', '\\tilde', '\\widetilde', '\\dot', '\\ddot',
    
#     # Blackboard and Fraktur Characters
#     '\\mathbb{R}', '\\mathbb{Q}', '\\mathbb{Z}', '\\mathbb{N}', '\\mathbb{C}', '\\mathbb{H}', '\\mathbb{P}',
#     '\\mathfrak{p}', '\\mathfrak{q}', '\\mathfrak{r}', '\\mathfrak{s}', '\\mathcal{A}', '\\mathcal{B}',
    
#     # # Geometry
#     # '\\triangle', '\\square', '\\blacksquare', '\\angle', '\\measuredangle', '\\perp', '\\parallel', '\\overrightarrow',
#     # '\\overleftarrow', '\\vec',
    
#     # # Logical Operators
#     # '\\vee', '\\wedge', '\\oplus', '\\ominus', '\\otimes', '\\oslash', '\\odot', '\\circledast',
#     # '\\bigoplus', '\\bigotimes', '\\bigodot',
    
#     # Delimiters
#     '\\lceil', '\\rceil', '\\lfloor', '\\rfloor', '\\langle', '\\rangle', '(', ')', '[', ']', '{', '}', '|',
    
#     # Text Formatting
#     '\\text', '\\mathrm', '\\mathbf', '\\mathit', '\\mathsf', '\\mathtt', '\\mathcal', '\\mathfrak',
    
#     # Numbers and Variables
#     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'x', 'y', 'i', 'j', '_'
    
#     # Probability and Statistics
#     '\\mathbb{P}', '\\text{E}', '\\text{Var}', '\\text{Cov}', '\\text{Pr}', '\\binom', '\\pmatrix',
    
#     # # Custom Symbols
#     # '\\therefore', '\\because', '\\vdash', '\\dashv', '\\models', '\\not\\models', '\\perp', '\\top', '\\bot',
# ])



# Your original functions exactly as they were
def iterative_segment_image(image, model, processor,density_threshold=FORMULA_THRESHOLD,  max_segments=10):
    """Your original iterative_segment_image function"""
    segment_count = 0
    kernel_size = 30
    iteration = 1
    formula_density = 1

    while segment_count < max_segments and kernel_size > 26 and formula_density > density_threshold:
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (250, kernel_size))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

        segment_count = num_labels - 1
        # print(f"Iteration {iteration}: Kernel Size {kernel_size}, Segments {segment_count}")

        formula_density = estimate_formula_density(closed, stats, model, processor)
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

def estimate_formula_density(image, stats, model, processor):
    """Your original estimate_formula_density function"""
    formula_symbols_count = 0
    total_chars = 0

    for i in range(1, len(stats)):
        x, y, w, h, area = stats[i]
        region = image[y:y + h, x:x + w]
        results = extract_text(image, model, processor, use_easyocr=True)
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



def modified_generate_latex_from_image(image, model, processor) -> str:
    """
    Generates LaTeX code from an image using the specified model and processor.

    Args:
        image (numpy.ndarray or PIL.Image.Image): The input image object.
        model: The trained LaTeX generation model.
        processor: The processor for preparing inputs for the model.

    Returns:
        str: Generated LaTeX code.
    """
    # Ensure the image is a PIL image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Ensure the image is in RGB mode
    image = image.convert("RGB")

    task_prompt = processor.tokenizer.bos_token
    decoder_input_ids = processor.tokenizer(
        task_prompt,
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids

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



def extract_text(image, model, processor, use_easyocr=False):
    """Extract text from an image using EasyOCR or Microsoft TrOCR"""
    if use_easyocr:
        results = modified_generate_latex_from_image(image, model, processor)
        extracted_text = results
    else:
        # Convert the image to RGB if needed
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        # Process the image using TrOCR
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return extracted_text

# def classify_text_type(extracted_text):
#     """Your original classify_text_type function"""
#     symbol_count = sum(1 for char in extracted_text if char in math_symbols)
#     total_chars = len(extracted_text)
#     if total_chars == 0:
#         return "Unknown"
#     symbol_density = symbol_count / total_chars
#     return "Formula" if symbol_density >= FORMULA_THRESHOLD_2 else "Text"

# Maximum weight assigned to any symbol
MAX_WEIGHT = max(math_weights.values())

def classify_text_type(extracted_text):
    # Split text into tokens and calculate total weight
    tokens = extracted_text.split()
    symbol_weight_sum = sum(math_weights.get(token, 0) for token in tokens)
    total_chars = len(extracted_text)
    
    if total_chars == 0:
        return "Unknown"
    
    # Calculate normalized density
    normalized_density = symbol_weight_sum / (total_chars * MAX_WEIGHT)
    
    return "Formula" if normalized_density >= FORMULA_THRESHOLD_2 else "Text"


def classify_image(image_path, model, processor, use_easyocr=True):
    """Classify the content of a binary-preprocessed image as Formula or Text"""
    preprocessed_image_path = preprocess_image(image_path)  # This now returns a path
    preprocessed_image = cv2.imread(preprocessed_image_path, cv2.IMREAD_GRAYSCALE)
    extracted_text = extract_text(preprocessed_image, model, processor, use_easyocr=use_easyocr)
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
    processor_trocr = TrOCRProcessor.from_pretrained('kazars24/trocr-base-handwritten-ru')
    model_trocr = VisionEncoderDecoderModel.from_pretrained('kazars24/trocr-base-handwritten-ru')
    return model, processor, processor_trocr, model_trocr

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

# def extract_text_from_image(image_path: str, processor: TrOCRProcessor, model: VisionEncoderDecoderModel) -> str:
#     # Load the image
#     image = Image.open(image_path).convert("RGB")

#     # Preprocess the image
#     pixel_values = processor(images=image, return_tensors="pt").pixel_values

#     # Generate text
#     generated_ids = model.generate(pixel_values)
#     extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

#     return extracted_text

def extract_text_from_image(image_path: str) -> str:
    """
    Extracts text from an image using Tesseract OCR.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Extracted text.
    """
    # Load the image
    image = Image.open(image_path)

    # Use Tesseract to extract text
    extracted_text = pytesseract.image_to_string(image, lang="eng")  # Specify language as needed

    return extracted_text.strip()


def convert_pdf_to_images(pdf_path):
    """
    Convert PDF pages to images and return list of image file paths.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        list: List of paths to the generated image files
    """
    image_paths = []
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path)

        # Save each page as an image
        for i, image in enumerate(images):
            # Create temporary file for the image
            image_path = create_temp_file(suffix=f"_page_{i+1}.png")
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
            print(f"Saved PDF page {i+1} as image: {image_path}")

    except Exception as e:
        st.error(f"Error converting PDF to images: {str(e)}")
        return []

    return image_paths

# Add these imports at the top of your main file
def main():
    st.title("Formula Detection and Recognition App")

    # Load models
    model, processor, processor_2, model_2 = load_models()

    # File upload with PDF support
    uploaded_file = st.file_uploader("Choose an image or PDF file", type=["png", "jpg", "jpeg", "pdf"])

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

            if not pdf_image_files:
                st.error("Failed to process the PDF file. Please try again.")
                return

            # Process each page
            for i, image_path in enumerate(pdf_image_files):
                st.subheader(f"Page {i+1}")

                # Convert the page image to black and white
                bw_image_path = convert_to_black_and_white(image_path)

                # Display original page image
                st.image(image_path, caption=f"Original - Page {i+1}")

                # Add a process button for each page
                if st.button(f"Process Page {i+1}"):
                    with st.spinner(f"Processing page {i+1}..."):
                        # Resize image
                        output_path = create_temp_file()
                        resize_image(bw_image_path, output_path, 1200, 2000)

                        # Load and process image
                        image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

                        # Segment image
                        segmented_image = iterative_segment_image(image, model, processor)

                        # Display segmented image
                        st.subheader(f"Segmented Image - Page {i+1}")
                        st.image(segmented_image)

                        # Process segments
                        L = {}
                        for seg_path in SEGMENT_IMAGE_SET:
                            classification = classify_image(seg_path, model, processor, use_easyocr=True)
                            print(f'Classification {seg_path}: {classification}')
                            L[seg_path] = classification

                        # Generate results
                        result = []
                        for seg_path in L:
                            if L[seg_path] == "Formula":
                                latex_expression = generate_latex_from_image(seg_path, model, processor)
                                formulas = latex_expression.split("\\newline")
                                for formula in formulas:
                                    result.append(f"\\begin{{equation}}\n{formula}\n\\end{{equation}}")
                            else:
                                text = extract_text_from_image(seg_path)
                                result.append(f"\n{text}\n")

                        # Display final output for this page
                        final_output = "\n%----\n".join(result)
                        final_output = final_output.replace("%", "\%")
                        final_output = final_output.replace("\%---", "%")
                        st.subheader(f"Output - Page {i+1}")
                        st.markdown(f"```\n{final_output}\n```", unsafe_allow_html=True)

            # Cleanup temporary files after processing all pages
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
                    resize_image(bw_image_path, output_path, 1000, 2000)

                    # Load and process image
                    image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

                    # Segment image using your original function
                    segmented_image = iterative_segment_image(image,)

                    # Display segmented image
                    st.subheader("Segmented Image")
                    st.image(segmented_image)

                    # Process segments using your original classification
                    L = {}
                    for img_path in SEGMENT_IMAGE_SET:
                        classification = classify_image(img_path, model, processor, use_easyocr=True)
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
                    # final_output.replace("%", "\%")
                    final_output = "\n".join(result)
                    # final_output.replace("\%", "%")
                    st.subheader("Complete Output")
                    col1, col2 = st.columns([4, 1])

                    # Display the text area in the first column
                    with col1:
                        st.markdown(f"```\n{final_output}\n```", unsafe_allow_html=True)
            # Cleanup after processing the single image
            cleanup_temp_files()


if __name__ == "__main__":
    main()
