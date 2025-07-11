import subprocess
import sys
import os
import platform
import venv

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
    "oauth2client==4.1.3"
]

def run_command(command, env=None):
    """Run a shell command and return success status."""
    try:
        subprocess.check_call(command, shell=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {command}\n{e}")
        return False

def create_virtual_env(env_name):
    """Create a virtual environment in the current directory."""
    if not os.path.exists(env_name):
        print(f"🔧 Creating virtual environment: {env_name}")
        venv.create(env_name, with_pip=True)
    else:
        print(f"✅ Virtual environment '{env_name}' already exists.")

def get_pip_path(env_name):
    """Return pip path for the virtual environment based on OS."""
    if platform.system() == "Windows":
        return os.path.join(env_name, "Scripts", "pip.exe")
    else:
        return os.path.join(env_name, "bin", "pip")

def get_python_path(env_name):
    """Return Python executable path in the virtual environment."""
    if platform.system() == "Windows":
        return os.path.join(env_name, "Scripts", "python.exe")
    else:
        return os.path.join(env_name, "bin", "python")

def install_packages(pip_path):
    """Install all required packages using pip."""
    print("📦 Installing dependencies...")
    for package in REQUIRED_PACKAGES:
        print(f"➡ Installing {package}")
        if not run_command(f'"{pip_path}" install {package}'):
            print(f"❌ Failed to install {package}. Exiting.")
            sys.exit(1)

def verify_versions(python_path):
    """Verify all installed package versions."""
    print("\n🔍 Verifying installed versions:")
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
    print("🚀 Starting Phase 1 Setup: Medical Assistant Environment")
    create_virtual_env(ENV_NAME)

    pip_path = get_pip_path(ENV_NAME)
    python_path = get_python_path(ENV_NAME)

    install_packages(pip_path)
    verify_versions(python_path)

    print("\n🎉 Phase 1 setup complete! You're good to go.")
    print(f"👉 To activate your environment, run:")
    if platform.system() == "Windows":
        print(f"   .\\{ENV_NAME}\\Scripts\\activate")
    else:
        print(f"   source {ENV_NAME}/bin/activate")

if __name__ == "__main__":
    main()
