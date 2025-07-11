# pdf_test_automation.py

import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import logging
from typing import List

# Import the robust text extraction function from our other script
try:
    from pdf_preprocessing import extract_text_robust
except ImportError:
    print("FATAL ERROR: pdf_preprocessing.py not found. Make sure it's in the same directory.")
    exit()

# --- Configuration ---
# !!! IMPORTANT !!!
# 1. UPDATE this with the exact name of your Google Sheet.
SHEET_NAME = "PDF_Test_Results" 
# 2. ENSURE 'credentials.json' is in the same directory as this script.
CREDS_FILE = 'credentials.json'
# 3. SET the column names as they appear in your sheet's header row.
PATH_COLUMN = 'pdf_path'
STATUS_COLUMN = 'test_status'
COUNT_COLUMN = 'character_count'
ERROR_COLUMN = 'error_details'

# Scopes required to access Google Sheets and Drive
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def authenticate_gsheet_client():
    """
    Authenticates with the Google Sheets API using service account credentials.

    Returns:
        gspread.Client: An authorized gspread client object.
        Returns None on failure.
    """
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPES)
        client = gspread.authorize(creds)
        logging.info("Successfully authenticated with Google Sheets API.")
        return client
    except FileNotFoundError:
        logging.error(f"FATAL: Credentials file not found at '{CREDS_FILE}'.")
        logging.error("Please follow the setup instructions to create and place the credentials file.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during Google API authentication: {e}")
        return None

def run_pdf_tests(worksheet):
    """
    Reads PDF paths from the worksheet, tests them, and logs results back.

    Args:
        worksheet (gspread.Worksheet): The worksheet object to interact with.
    """
    try:
        logging.info("Fetching data from Google Sheet...")
        df = pd.DataFrame(worksheet.get_all_records())
        if PATH_COLUMN not in df.columns:
            logging.error(f"FATAL: The required column '{PATH_COLUMN}' was not found in the sheet.")
            logging.error("Please ensure your sheet has the correct headers.")
            return
        logging.info(f"Found {len(df)} PDF paths to process.")
    except Exception as e:
        logging.error(f"Failed to read data from the worksheet: {e}")
        return

    results = []
    for index, row in df.iterrows():
        pdf_path = row[PATH_COLUMN]
        status = 'FAIL'
        char_count = 0
        error_msg = ''

        logging.info(f"Processing row {index + 2}: {pdf_path}")

        try:
            if not pdf_path:
                error_msg = "PDF path is empty."
            elif not os.path.exists(pdf_path):
                error_msg = f"File not found at path: {pdf_path}"
            else:
                extracted_text = extract_text_robust(pdf_path)
                
                if extracted_text:
                    char_count = len(extracted_text)
                    if char_count > 100:
                        status = 'PASS'
                        error_msg = ''
                    else:
                        error_msg = "Extracted text is too short (<= 100 chars)."
                else:
                    error_msg = "Text extraction failed; function returned None."

        except Exception as e:
            error_msg = f"An unexpected script error occurred: {e}"
            logging.error(f"Error processing {pdf_path}: {error_msg}")

        results.append({
            STATUS_COLUMN: status,
            COUNT_COLUMN: char_count,
            ERROR_COLUMN: error_msg
        })
        logging.info(f"Result for {pdf_path}: {status}, Chars: {char_count}")

    # Update the DataFrame with results
    results_df = pd.DataFrame(results)
    df[STATUS_COLUMN] = results_df[STATUS_COLUMN]
    df[COUNT_COLUMN] = results_df[COUNT_COLUMN]
    df[ERROR_COLUMN] = results_df[ERROR_COLUMN]

    # Update the Google Sheet
    try:
        logging.info("Updating Google Sheet with all results...")
        # Add +1 to the column index because gspread's find function is 1-based.
        status_col_index = df.columns.get_loc(STATUS_COLUMN) + 1
        count_col_index = df.columns.get_loc(COUNT_COLUMN) + 1
        error_col_index = df.columns.get_loc(ERROR_COLUMN) + 1
        
        # Prepare data for batch update
        update_range = f'{gspread.utils.rowcol_to_a1(2, status_col_index)}:{gspread.utils.rowcol_to_a1(len(df) + 1, error_col_index)}'
        update_values = df[[STATUS_COLUMN, COUNT_COLUMN, ERROR_COLUMN]].values.tolist()

        worksheet.update(update_range, update_values)
        logging.info("Google Sheet update complete.")
    except Exception as e:
        logging.error(f"Failed to update the Google Sheet. Error: {e}")
        logging.info("Results were not saved to the cloud.")


def main():
    """Main function to orchestrate the PDF test automation."""
    logging.info("--- Starting PDF Test Automation Script ---")
    
    client = authenticate_gsheet_client()
    if not client:
        return # Stop execution if authentication fails

    try:
        spreadsheet = client.open(SHEET_NAME)
        worksheet = spreadsheet.sheet1 # Get the first sheet
        logging.info(f"Successfully opened worksheet '{worksheet.title}' in spreadsheet '{SHEET_NAME}'.")
    except gspread.exceptions.SpreadsheetNotFound:
        logging.error(f"FATAL: Spreadsheet named '{SHEET_NAME}' not found.")
        logging.error("Please check the sheet name and ensure it has been shared with the service account email.")
        return
    except Exception as e:
        logging.error(f"An error occurred while opening the sheet: {e}")
        return

    run_pdf_tests(worksheet)
    logging.info("--- Script Finished ---")


if __name__ == '__main__':
    # Create a dummy PDF and sheet instructions for first-time use
    if not os.path.exists("sample_medical_report.pdf"):
        logging.warning("Creating a dummy PDF 'sample_medical_report.pdf' for testing purposes.")
        logging.warning("Please add its path './sample_medical_report.pdf' to your Google Sheet.")
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            c = canvas.Canvas("sample_medical_report.pdf", pagesize=letter)
            text = c.beginText(72, 800)
            text.textLine("This is a sample PDF document created for test automation.")
            text.textLine("It contains more than one hundred characters to ensure that the")
            text.textLine("text extraction test case will result in a PASS status.")
            text.textLine("This helps verify that the entire workflow, from file access")
            text.textLine("to Google Sheets API communication, is functioning correctly.")
            c.drawText(text)
            c.save()
        except ImportError:
            logging.error("Please install reportlab (`pip install reportlab`) to create a sample PDF.")
        except Exception as e:
            logging.error(f"Could not create dummy PDF: {e}")
            
    main()