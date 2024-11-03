# Document Processing Application

A Streamlit-based web application that processes PDF documents and images, performing segmentation analysis and generating structured output. The app supports both PDF documents with multiple pages and individual image files.

## 🌟 Features

- **PDF Processing**:
  - Convert PDF files to images
  - Page-by-page navigation
  - Individual page processing
  - Image segmentation visualization
  
- **Image Processing**:
  - Support for various image formats
  - Image segmentation analysis
  - Structured output generation

- **User Interface**:
  - Clean and intuitive interface
  - Progress indicators for long operations
  - Copy-to-clipboard functionality
  - Preview of original and processed images

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit
pip install Pillow
# Add other required packages based on your model and processor dependencies
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 💻 Usage

1. Launch the application using the command above
2. Upload a file using the file uploader:
   - For PDFs: Navigate through pages using Previous/Next buttons
   - For Images: Direct processing
3. Click "Process" to analyze the current page/image
4. View the segmented image and structured output
5. Copy results to clipboard using the Copy button

## 🔧 Configuration

The application uses session state to maintain page navigation and clipboard content. Key configurations:

- `st.session_state.current_page`: Tracks current PDF page
- `SEGMENT_IMAGE_SET`: Manages processed image segments
- Temporary file cleanup is handled automatically

## 🛠️ Technical Details

The application consists of two main processing paths:

### PDF Processing
- Converts PDF pages to images
- Provides navigation between pages
- Processes individual pages on demand

### Image Processing
- Handles direct image file uploads
- Performs segmentation analysis
- Generates structured output

## 📝 Output Format

The application generates structured output with the following format:
```
segment_1
%----
segment_2
%----
segment_3
```
Note: % symbols are escaped in the output for compatibility.

## ⚠️ Dependencies

- Streamlit
- PDF processing library (specify your choice)
- Image processing libraries
- Your custom model and processor

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- List any third-party tools or libraries you're using
- Credit any inspirations or resources used
