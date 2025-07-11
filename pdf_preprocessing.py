import os
import sys
import PyPDF2
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_robust(pdf_file: str) -> str | None:
    """
    Extracts text from a PDF file robustly.

    It first attempts to use pdfplumber for extraction. If the extracted text
    is less than 100 characters or if pdfplumber fails, it falls back to PyPDF2.
    If both methods fail, it returns None.

    Args:
        pdf_file: The path to the PDF file.

    Returns:
        The extracted text as a string, or None if extraction fails.
    """
    text = ""
    try:
        # Attempt to open with pdfplumber
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        logging.info(f"Successfully extracted text with pdfplumber from {pdf_file}")

        if len(text) < 100:
            logging.warning(f"Extracted text with pdfplumber is short (< 100 chars) for {pdf_file}. Falling back to PyPDF2.")
            raise ValueError("Text too short")

    except Exception as e:
        logging.error(f"pdfplumber failed for {pdf_file}: {e}. Falling back to PyPDF2.")
        text = ""  # Reset text before trying PyPDF2
        try:
            with open(pdf_file, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text:
                logging.info(f"Successfully extracted text with PyPDF2 from {pdf_file}")
            else:
                logging.warning(f"PyPDF2 extracted no text from {pdf_file}.")
                return None

        except Exception as e_pypdf:
            logging.error(f"Both pdfplumber and PyPDF2 failed to extract text from {pdf_file}.")
            logging.error(f"PyPDF2 error: {e_pypdf}")
            return None

    return text.strip()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Splits a long text into smaller chunks with a specified overlap.

    Args:
        text: The text to be chunked.
        chunk_size: The desired character length of each chunk.
        overlap: The number of characters to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if start + overlap >= len(text):
             # Add the last remaining part of the text
            if len(text) > end:
                chunks.append(text[end:])
            break # Exit loop to avoid infinite loop on last chunk
            
    return chunks

def store_in_chroma(chunks: list[str], filename: str):
    """
    Stores text chunks and their embeddings in a local ChromaDB collection.

    Args:
        chunks: A list of text chunks to be stored.
        filename: The name of the source file, used for the collection name.
    """
    if not chunks:
        logging.warning("No chunks to store. Aborting ChromaDB storage.")
        return

    db_path = "./chroma_db"
    collection_name = os.path.splitext(os.path.basename(filename))[0].replace(" ", "_")

    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=db_path)

        # Initialize the embedding model
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logging.error(f"Failed to load SentenceTransformer model: {e}")
            return
            
        # Create or get the collection
        collection = client.get_or_create_collection(name=collection_name)

        # Prepare documents, metadatas, and ids
        documents = chunks
        metadatas = [{"source": filename, "chunk_number": i} for i, _ in enumerate(chunks)]
        ids = [f"{filename}_chunk_{i}" for i, _ in enumerate(chunks)]

        # Generate embeddings and store in ChromaDB
        try:
            embeddings = model.encode(documents, show_progress_bar=True).tolist()
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logging.info(f"Successfully stored {len(chunks)} chunks in ChromaDB collection '{collection_name}' at {db_path}")
        except Exception as e:
            logging.error(f"Failed to generate embeddings or store in ChromaDB: {e}")

    except Exception as e:
        logging.error(f"Failed to initialize or interact with ChromaDB: {e}")


if __name__ == '__main__':
    # --- Test Section ---

    # Create a dummy PDF for testing
    sample_pdf_filename = "sample_medical_report.pdf"
    
    # For a real-world scenario, you would have an actual PDF file.
    # We will create a simple one here for demonstration if it doesn't exist.
    if not os.path.exists(sample_pdf_filename):
        logging.info(f"Creating a dummy PDF: {sample_pdf_filename}")
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            c = canvas.Canvas(sample_pdf_filename, pagesize=letter)
            width, height = letter
            
            # Page 1
            text = c.beginText(72, height - 72)
            text.setFont("Helvetica", 10)
            text.textLine("Patient Name: John Doe")
            text.textLine("Date of Birth: 1985-05-20")
            text.textLine("Medical Record Number: 123456")
            text.textLine("-" * 50)
            text.textLine("Chief Complaint: Persistent cough and shortness of breath.")
            text.textLine("History of Present Illness: The patient is a 38-year-old male with a 2-month history of a")
            text.textLine("productive cough, occasionally blood-streaked, accompanied by progressive dyspnea.")
            text.textLine("He also reports intermittent low-grade fevers and a 10-pound weight loss over the past 6 weeks.")
            text.textLine("Past Medical History: Non-contributory. No history of chronic illnesses.")
            c.drawText(text)
            c.showPage()
            
            # Page 2
            text = c.beginText(72, height - 72)
            text.setFont("Helvetica", 10)
            text.textLine("Chest X-Ray Findings:")
            text.textLine("A 3 cm cavitary lesion is noted in the right upper lobe. Associated parenchymal")
            text.textLine("opacities are present. The cardiomediastinal silhouette is within normal limits.")
            text.textLine("Impression: Findings are highly suggestive of pulmonary tuberculosis.")
            text.textLine("Recommendation: Sputum for Acid-Fast Bacilli (AFB) smear and culture is recommended for confirmation.")
            c.drawText(text)
            c.save()
            logging.info("Dummy PDF created successfully.")
        except ImportError:
            logging.error("ReportLab is not installed. Cannot create a dummy PDF.")
            logging.info("Please create a file named 'sample_medical_report.pdf' manually to run the test.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"An error occurred while creating the dummy PDF: {e}")
            sys.exit(1)


    # 1. Extract text from the sample PDF
    logging.info("\n--- Step 1: Extracting Text ---")
    extracted_text = extract_text_robust(sample_pdf_filename)
    if extracted_text:
        print("Extracted Text (first 300 chars):")
        print(extracted_text[:300] + "...")
    else:
        logging.error("Text extraction failed. Exiting.")
        sys.exit(1)

    # 2. Chunk the extracted text
    logging.info("\n--- Step 2: Chunking Text ---")
    text_chunks = chunk_text(extracted_text)
    if text_chunks:
        print(f"Text divided into {len(text_chunks)} chunks.")
        print("First chunk:")
        print(text_chunks[0])
    else:
        logging.warning("No chunks were created from the extracted text.")

    # 3. Store chunks in ChromaDB
    logging.info("\n--- Step 3: Storing in ChromaDB ---")
    store_in_chroma(text_chunks, sample_pdf_filename)

    # Verify storage by querying ChromaDB (optional)
    try:
        logging.info("\n--- Verification: Querying ChromaDB ---")
        client = chromadb.PersistentClient(path="./chroma_db")
        collection_name = os.path.splitext(os.path.basename(sample_pdf_filename))[0].replace(" ", "_")
        collection = client.get_collection(name=collection_name)
        
        query_result = collection.query(
            query_texts=["What are the findings of the chest x-ray?"],
            n_results=1
        )
        print("Query Result for 'What are the findings of the chest x-ray?':")
        if query_result and query_result['documents']:
            print(query_result['documents'][0])
        else:
            print("No relevant documents found.")
            
    except Exception as e:
        logging.error(f"Could not query ChromaDB for verification: {e}")