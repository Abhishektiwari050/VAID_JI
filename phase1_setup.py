"""
Phase 1 Setup Script for VAID JI Medical Research Assistant
Automates virtual environment creation and dependency installation
"""

import subprocess
import sys
import os
import platform
import venv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define your environment name
ENV_NAME = "medical_assistant_env"

# List of required packages with exact versions
REQUIRED_PACKAGES = [
    "streamlit==1.28.1",
    "PyPDF2==3.0.1",
    "pdfplumber==0.10.3",
    "pandas==2.1.4",
    "numpy==1.24.3",
    "streamlit-extras==0.3.5",
    "python-dotenv==1.0.0",
    "sentence-transformers==2.2.2",
    "chromadb==0.4.22",
    "gspread==5.12.0",
    "oauth2client==4.1.3",
    "pytesseract==0.3.10",
    "reportlab==4.0.9",
    "langchain==0.0.350"
]

def run_command(command: str, env: dict = None) -> bool:
    """
    Run a shell command and return success status.
    
    Args:
        command: The shell command to execute
        env: Optional environment variables
        
    Returns:
        bool: True if command succeeded, False otherwise
    """
    try:
        subprocess.check_call(command, shell=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running command: {command}\n{e}")
        return False


def create_virtual_env(env_name: str):
    """
    Create a virtual environment in the current directory.
    
    Args:
        env_name: Name of the virtual environment to create
    """
    if not os.path.exists(env_name):
        logger.info(f"Creating virtual environment: {env_name}")
        venv.create(env_name, with_pip=True)
    else:
        logger.info(f"Virtual environment '{env_name}' already exists.")


def get_pip_path(env_name: str) -> str:
    """
    Return pip path for the virtual environment based on OS.
    
    Args:
        env_name: Name of the virtual environment
        
    Returns:
        str: Path to pip executable
    """
    if platform.system() == "Windows":
        return os.path.join(env_name, "Scripts", "pip.exe")
    else:
        return os.path.join(env_name, "bin", "pip")


def get_python_path(env_name: str) -> str:
    """
    Return Python executable path in the virtual environment.
    
    Args:
        env_name: Name of the virtual environment
        
    Returns:
        str: Path to Python executable
    """
    if platform.system() == "Windows":
        return os.path.join(env_name, "Scripts", "python.exe")
    else:
        return os.path.join(env_name, "bin", "python")


def install_packages(pip_path: str):
    """
    Install all required packages using pip.
    
    Args:
        pip_path: Path to pip executable
    """
    logger.info("Installing dependencies...")
    for package in REQUIRED_PACKAGES:
        logger.info(f"Installing {package}")
        if not run_command(f'"{pip_path}" install {package}'):
            logger.error(f"Failed to install {package}. Exiting.")
            sys.exit(1)

def verify_versions(python_path: str):
    """
    Verify all installed package versions.
    
    Args:
        python_path: Path to Python executable
    """
    logger.info("\n Verifying installed versions:")
    version_check_code = """
import streamlit, PyPDF2, pdfplumber, pandas, numpy, streamlit_extras, dotenv, sentence_transformers, chromadb, gspread, oauth2client

print("✅ streamlit:", streamlit.__version__)
print("✅ PyPDF2:", PyPDF2.__version__)
print("✅ pdfplumber:", pdfplumber.__version__)
print("✅ pandas:", pandas.__version__)
print("✅ numpy:", numpy.__version__)
print("✅ streamlit-extras:", streamlit_extras.__version__)
print("✅ python-dotenv:", dotenv.__version__)
print("✅ sentence-transformers:", sentence_transformers.__version__)
print("✅ chromadb:", chromadb.__version__)
print("✅ gspread:", gspread.__version__)
print("✅ oauth2client:", oauth2client.__version__)
"""
    run_command(f'"{python_path}" -c "{version_check_code.strip()}"')


def main():
    """Main setup function"""
    logger.info("Starting Phase 1 Setup: Medical Assistant Environment")
    create_virtual_env(ENV_NAME)

    pip_path = get_pip_path(ENV_NAME)
    python_path = get_python_path(ENV_NAME)

    install_packages(pip_path)
    verify_versions(python_path)

    logger.info("\n Phase 1 setup complete! You're good to go.")
    logger.info(f"To activate your environment, run:")
    if platform.system() == "Windows":
        logger.info(f"   .\\{ENV_NAME}\\Scripts\\activate")
    else:
        logger.info(f"   source {ENV_NAME}/bin/activate")


if __name__ == "__main__":
    main()
