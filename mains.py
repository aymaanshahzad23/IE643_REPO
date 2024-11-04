
# def main():
#     st.title("Formula Detection and Recognition App")

#     # Load models
#     model, processor = load_models()

#     # Initialize session state for temp files and processing
#     if 'temp_files' not in st.session_state:
#         st.session_state.temp_files = []
#     if 'SEGMENT_IMAGE_SET' not in st.session_state:
#         st.session_state.SEGMENT_IMAGE_SET = []

#     # File upload
#     uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "pdf"])

#     if uploaded_file is not None:
#         # Check if the uploaded file is a PDF
#         if uploaded_file.type == "application/pdf":
#             # Create a temporary file for the PDF
#             pdf_temp_file = create_temp_file(suffix=".pdf")
#             with open(pdf_temp_file, "wb") as f:
#                 f.write(uploaded_file.getvalue())

#             # Convert PDF pages to images
#             pdf_image_files = convert_pdf_to_images(pdf_temp_file)
#             st.session_state.SEGMENT_IMAGE_SET.extend(pdf_image_files)  # Add the converted images to the segment list

#             # Display each image with a processing button
#             for image_path in st.session_state.SEGMENT_IMAGE_SET:
#                 # Convert the uploaded image to black and white
#                 bw_image_path = convert_to_black_and_white(image_path)

#                 # Display original image
#                 st.subheader("Original Image")
#                 st.image(image_path)

#                 # Process image
#                 if st.button(f"Process Image: {image_path}"):
#                     with st.spinner("Processing image..."):
#                         # Resize image
#                         output_path = create_temp_file()
#                         resize_image(bw_image_path, output_path, 900, 2000)

#                         # Load and process image
#                         image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

#                         # Segment image using your original function
#                         segmented_image = iterative_segment_image(image)

#                         # Display segmented image
#                         st.subheader("Segmented Image")
#                         st.image(segmented_image)

#                         # Process segments using your original classification
#                         L = {}
#                         for img_path in st.session_state.SEGMENT_IMAGE_SET:
#                             classification = classify_image(img_path, use_easyocr=True)
#                             print(f'Classification {img_path}: {classification}')
#                             L[img_path] = classification

#                         # Generate results as in your original code
#                         result = []
#                         for img_path in L:
#                             if L[img_path] == "Formula":
#                                 latex_expression = generate_latex_from_image(img_path, model, processor)
#                                 formulas = latex_expression.split("\\newline")
#                                 for formula in formulas:
#                                     result.append(f"\\begin{{equation}}\n{formula}\n\\end{{equation}}")
#                             else:
#                                 text = extract_text_from_image(img_path)
#                                 result.append(f"\n{text}\n")

#                         # Display final output as in your original code
#                         final_output = "\n%----\n".join(result)
#                         final_output.replace("%", "\%")
#                         st.subheader("Complete Output")
#                         col1, col2 = st.columns([4, 1])

#                         # Display the text area in the first column
#                         with col1:
#                             st.text_area("", value=final_output, height=500, key="output_area")

#             # Cleanup after all images are processed
#             # cleanup_temp_files()

#         else:
#             # Handle non-PDF image upload (JPEG/PNG)
#             input_path = create_temp_file()
#             with open(input_path, "wb") as f:
#                 f.write(uploaded_file.getvalue())

#             # Convert the uploaded image to black and white
#             bw_image_path = convert_to_black_and_white(input_path)

#             # Display original image
#             st.subheader("Original Image")
#             st.image(input_path)

#             # Process image
#             if st.button("Process Image"):
#                 with st.spinner("Processing image..."):
#                     # Resize image
#                     output_path = create_temp_file()
#                     resize_image(bw_image_path, output_path, 900, 2000)

#                     # Load and process image
#                     image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

#                     # Segment image using your original function
#                     segmented_image = iterative_segment_image(image)

#                     # Display segmented image
#                     st.subheader("Segmented Image")
#                     st.image(segmented_image)

#                     # Process segments using your original classification
#                     L = {}
#                     for img_path in st.session_state.SEGMENT_IMAGE_SET:
#                         classification = classify_image(img_path, use_easyocr=True)
#                         print(f'Classification {img_path}: {classification}')
#                         L[img_path] = classification

#                     # Generate results as in your original code
#                     result = []
#                     for img_path in L:
#                         if L[img_path] == "Formula":
#                             latex_expression = generate_latex_from_image(img_path, model, processor)
#                             formulas = latex_expression.split("\\newline")
#                             for formula in formulas:
#                                 result.append(f"\\begin{{equation}}\n{formula}\n\\end{{equation}}")
#                         else:
#                             text = extract_text_from_image(img_path)
#                             result.append(f"\n{text}\n")

#                     # Display final output as in your original code
#                     final_output = "\n%----\n".join(result)
#                     final_output.replace("%", "\%")
#                     st.subheader("Complete Output")
#                     col1, col2 = st.columns([4, 1])

#                     # Display the text area in the first column
#                     with col1:
#                         st.text_area("", value=final_output, height=500, key="output_area")
#             # Cleanup after single image processing
#             cleanup_temp_files()

if __name__ == "__main__":
    main()

# def main():
#     st.title("Formula Detection and Recognition App")

#     # Load models
#     model, processor = load_models()

#     # Initialize session state for temp files and processing
#     if 'temp_files' not in st.session_state:
#         st.session_state.temp_files = []
#     if 'SEGMENT_IMAGE_SET' not in st.session_state:
#         st.session_state.SEGMENT_IMAGE_SET = []
#     if 'processed_images' not in st.session_state:
#         st.session_state.processed_images = {}

#     # File upload
#     uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "pdf"])

#     if uploaded_file is not None:
#         # Check if the uploaded file is a PDF
#         if uploaded_file.type == "application/pdf":
#             # Create a temporary file for the PDF
#             pdf_temp_file = create_temp_file(suffix=".pdf")
#             with open(pdf_temp_file, "wb") as f:
#                 f.write(uploaded_file.getvalue())

#             # Convert PDF pages to images
#             pdf_image_files = convert_pdf_to_images(pdf_temp_file)
#             st.session_state.SEGMENT_IMAGE_SET.extend(pdf_image_files)  # Add the converted images to the segment list

#             # Display each image with a processing button
#             for image_path in st.session_state.SEGMENT_IMAGE_SET:
#                 # If the image has not been processed
#                 if image_path not in st.session_state.processed_images:
#                     # Convert the uploaded image to black and white
#                     bw_image_path = convert_to_black_and_white(image_path)

#                     # Display original image
#                     st.subheader("Original Image")
#                     st.image(image_path)

#                     # Process image
#                     if st.button(f"Process Image: {image_path}"):
#                         with st.spinner("Processing image..."):
#                             # Resize image
#                             output_path = create_temp_file()
#                             resize_image(bw_image_path, output_path, 900, 2000)

#                             # Load and process image
#                             image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

#                             # Segment image using your original function
#                             segmented_image = iterative_segment_image(image)

#                             # Display segmented image
#                             st.subheader("Segmented Image")
#                             st.image(segmented_image)

#                             # Process segments using your original classification
#                             L = {}
#                             for img_path in st.session_state.SEGMENT_IMAGE_SET:
#                                 classification = classify_image(img_path, use_easyocr=True)
#                                 print(f'Classification {img_path}: {classification}')
#                                 L[img_path] = classification

#                             # Generate results
#                             result = []
#                             for img_path in L:
#                                 if L[img_path] == "Formula":
#                                     latex_expression = generate_latex_from_image(img_path, model, processor)
#                                     formulas = latex_expression.split("\\newline")
#                                     for formula in formulas:
#                                         result.append(f"\\begin{{equation}}\n{formula}\n\\end{{equation}}")
#                                 else:
#                                     text = extract_text_from_image(img_path)
#                                     result.append(f"\n{text}\n")

#                             # Store the results in session state
#                             final_output = "\n%----\n".join(result)
#                             final_output.replace("%", "\%")
#                             st.subheader("Complete Output")
#                             col1, col2 = st.columns([4, 1])

#                             # Display the text area in the first column
#                             with col1:
#                                 st.text_area("", value=final_output, height=500, key="output_area")

#                             # Mark the image as processed
#                             st.session_state.processed_images[image_path] = final_output

#         else:
#             # Handle non-PDF image upload (JPEG/PNG)
#             input_path = create_temp_file()
#             with open(input_path, "wb") as f:
#                 f.write(uploaded_file.getvalue())

#             # Convert the uploaded image to black and white
#             bw_image_path = convert_to_black_and_white(input_path)

#             # Display original image
#             st.subheader("Original Image")
#             st.image(input_path)

#             # Process image
#             if st.button("Process Image"):
#                 with st.spinner("Processing image..."):
#                     # Resize image
#                     output_path = create_temp_file()
#                     resize_image(bw_image_path, output_path, 900, 2000)

#                     # Load and process image
#                     image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

#                     # Segment image using your original function
#                     segmented_image = iterative_segment_image(image)

#                     # Display segmented image
#                     st.subheader("Segmented Image")
#                     st.image(segmented_image)

#                     # Process segments using your original classification
#                     L = {}
#                     for img_path in st.session_state.SEGMENT_IMAGE_SET:
#                         classification = classify_image(img_path, use_easyocr=True)
#                         print(f'Classification {img_path}: {classification}')
#                         L[img_path] = classification

#                     # Generate results
#                     result = []
#                     for img_path in L:
#                         if L[img_path] == "Formula":
#                             latex_expression = generate_latex_from_image(img_path, model, processor)
#                             formulas = latex_expression.split("\\newline")
#                             for formula in formulas:
#                                 result.append(f"\\begin{{equation}}\n{formula}\n\\end{{equation}}")
#                         else:
#                             text = extract_text_from_image(img_path)
#                             result.append(f"\n{text}\n")

#                     # Store the results in session state
#                     final_output = "\n%----\n".join(result)
#                     final_output.replace("%", "\%")
#                     st.subheader("Complete Output")
#                     col1, col2 = st.columns([4, 1])

#                     # Display the text area in the first column
#                     with col1:
#                         st.text_area("", value=final_output, height=500, key="output_area")
#             # Cleanup after single image processing
#             cleanup_temp_files()

# if __name__ == "__main__":
#     main()
