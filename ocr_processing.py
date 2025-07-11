# ocr_processing.py

import os
import pytesseract
import pdfplumber
import logging
from PIL import Image

# --- Configuration ---
# If Tesseract is not in your system's PATH, uncomment and set the following line:
# For Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For Linux/macOS, it's usually in the PATH and this is not needed.

DB_PATH = "./chroma_db"  # Changed to relative path for portability
# To use the original path, set: DB_PATH = r"D:\Projects\VAID_JI\chroma_db"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Import reusable functions from pdf_preprocessing.py ---
try:
    from pdf_preprocessing import chunk_text, store_in_chroma
    PDF_PREPROCESSING_AVAILABLE = True
except ImportError:
    PDF_PREPROCESSING_AVAILABLE = False
    logging.critical("FATAL ERROR: pdf_preprocessing.py not found.")
    logging.critical("This script cannot function without it. Please place it in the same directory.")
    # Define dummy functions to prevent immediate script crash on load
    def chunk_text(text, **kwargs): return []
    def store_in_chroma(chunks, filename): pass


def extract_text_with_ocr(pdf_path: str) -> str | None:
    """
    Extracts text from a PDF using a hybrid approach.
    It processes each page, attempts to extract text directly, and if a page
    yields minimal text (indicating a scanned image), it performs OCR on that page.

    Args:
        pdf_path: The full path to the PDF file.

    Returns:
        A string containing all extracted text, or None if the file cannot be processed.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"File not found: {pdf_path}")
        return None

    full_text = []
    logging.info(f"Starting hybrid text/OCR extraction for '{os.path.basename(pdf_path)}'.")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                logging.info(f"Processing page {i + 1}/{len(pdf.pages)}...")

                # 1. Attempt direct text extraction
                text = page.extract_text()

                # 2. Decide whether to use OCR
                # Heuristic: If text is very short or None, treat as a scanned/image-based page.
                if text and len(text.strip()) > 50:
                    logging.info(f"Page {i + 1}: Extracted text directly.")
                    full_text.append(text)
                else:
                    logging.warning(f"Page {i + 1}: Minimal text found. Applying OCR...")
                    try:
                        # Convert page to a high-resolution image for better OCR accuracy
                        # 300 DPI is a good standard for OCR.
                        page_image = page.to_image(resolution=300).original

                        # Use pytesseract to perform OCR on the image
                        ocr_text = pytesseract.image_to_string(page_image, lang='eng')
                        
                        if ocr_text:
                            logging.info(f"Page {i + 1}: Successfully extracted text via OCR.")
                            full_text.append(ocr_text)
                        else:
                            logging.warning(f"Page {i + 1}: OCR did not detect any text.")

                    except pytesseract.TesseractNotFoundError:
                        logging.error("Tesseract is not installed or not in your PATH.")
                        logging.error("Please install Tesseract and configure the path if necessary.")
                        return None # Abort if Tesseract is not found
                    except Exception as e:
                        logging.error(f"Failed to perform OCR on page {i + 1}: {e}")
                        
        return "\n".join(full_text).strip()

    except Exception as e:
        logging.error(f"An error occurred while processing the PDF file '{pdf_path}': {e}")
        return None


if __name__ == '__main__':
    if not PDF_PREPROCESSING_AVAILABLE:
        exit()
        
    logging.info("--- Starting OCR Processing Test Script ---")

    # --- Create a Sample Scanned PDF for Testing ---
    # This simulates a real-world scanned document by creating a PDF that contains only an image.
    scanned_pdf_filename = "sample_scanned_medical_report.pdf"
    if not os.path.exists(scanned_pdf_filename):
        logging.info(f"Generating a sample scanned PDF: '{scanned_pdf_filename}'")
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.utils import ImageReader

            # Step 1: Create a temporary text-based PDF to generate an image from.
            temp_text_pdf = "temp_text_for_image.pdf"
            c = canvas.Canvas(temp_text_pdf, pagesize=letter)
            text = c.beginText(72, 800)
            text.setFont("Helvetica", 12)
            text.textLine("Patient: Jane Smith, MRN: 78910")
            text.textLine("Diagnosis: Suspected Cholecystitis")
            text.textLine("--- Imaging Report ---")
            text.textLine("Ultrasound of the abdomen reveals a thickened gallbladder wall (5mm).")
            text.textLine("Pericholecystic fluid is noted. The common bile duct is not dilated.")
            text.textLine("Impression: Findings are consistent with acute cholecystitis.")
            c.drawText(text)
            c.save()

            # Step 2: Convert the first page of the temp PDF to a Pillow Image
            with pdfplumber.open(temp_text_pdf) as temp_pdf:
                page_image = temp_pdf.pages[0].to_image(resolution=200).original
                # Save as a temporary image file
                temp_image_path = "temp_page_image.png"
                page_image.save(temp_image_path)
            
            # Step 3: Create the final "scanned" PDF by embedding the image.
            c = canvas.Canvas(scanned_pdf_filename, pagesize=letter)
            width, height = letter
            # Draw the image to fill the page
            c.drawImage(ImageReader(temp_image_path), 0, 0, width=width, height=height)
            c.save()

            # Step 4: Clean up temporary files
            os.remove(temp_text_pdf)
            os.remove(temp_image_path)
            logging.info("Sample scanned PDF created successfully.")

        except Exception as e:
            logging.error(f"Failed to create a sample scanned PDF. Please ensure 'reportlab' is installed. Error: {e}")
            # Clean up if files were partially created
            if os.path.exists(scanned_pdf_filename): os.remove(scanned_pdf_filename)
            if os.path.exists("temp_text_for_image.pdf"): os.remove("temp_text_for_image.pdf")
            if os.path.exists("temp_page_image.png"): os.remove("temp_page_image.png")
            exit()
    else:
        logging.info(f"Using existing sample PDF: '{scanned_pdf_filename}'")

    # 1. Extract text from the sample scanned PDF
    logging.info("\n--- Step 1: Extracting Text with OCR ---")
    extracted_text = extract_text_with_ocr(scanned_pdf_filename)
    
    if extracted_text:
        print("\n--- OCR Extraction Result ---")
        print(extracted_text)
        print("---------------------------\n")

        # 2. Chunk the extracted text using the function from pdf_preprocessing.py
        logging.info("\n--- Step 2: Chunking Text ---")
        text_chunks = chunk_text(extracted_text, chunk_size=400, overlap=100)
        if text_chunks:
            print(f"Text divided into {len(text_chunks)} chunks.")
            print("First chunk:")
            print(text_chunks[0])
        else:
            logging.warning("No chunks were created from the extracted text.")

        # 3. Store chunks in ChromaDB using the function from pdf_preprocessing.py
        logging.info("\n--- Step 3: Storing in ChromaDB ---")
        # Override the collection name to be specific to the OCR'd doc
        # store_in_chroma expects a list of chunks and the original filename
        store_in_chroma(text_chunks, scanned_pdf_filename)
    else:
        logging.error("OCR text extraction failed. Cannot proceed with chunking and storing.")

    logging.info("\n--- OCR Processing Test Script Finished ---")