# scripts/check_environment.py
# G-MASS environment validation script
#
# Run this to verify the project environment is ready for use.
#
# Usage:
#   python scripts/check_environment.py

import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig


def module_available(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


print("\n" + "=" * 60)
print("  G-MASS Project - Environment Setup Check")
print("=" * 60 + "\n")

errors = []

print("Checking Python version...")
major, minor = sys.version_info.major, sys.version_info.minor
if major == 3 and minor >= 10:
    print(f"  OK Python {major}.{minor}\n")
else:
    print(f"  FAIL Python {major}.{minor} - Need Python 3.10 or higher")
    print("    Download from: python.org\n")
    errors.append("Python version too old")

print("Checking installed packages...")
packages = {
    "openai": "pip install -r requirements.txt",
    "google.genai": "pip install -r requirements.txt",
    "huggingface_hub": "pip install -r requirements.txt",
    "requests": "pip install -r requirements.txt",
    "dotenv": "pip install -r requirements.txt",
    "yaml": "pip install -r requirements.txt",
    "numpy": "pip install -r requirements.txt",
    "fasttext": "pip install -r requirements.txt",
    "openpyxl": "pip install -r requirements.txt",
    "pandas": "pip install -r requirements.txt",
    "jsonlines": "pip install -r requirements.txt",
    "pytest": "pip install -r requirements.txt",
}

for pkg, install_cmd in packages.items():
    if module_available(pkg):
        print(f"  OK {pkg}")
    else:
        print(f"  FAIL {pkg} - Run: {install_cmd}")
        errors.append(f"Missing package: {pkg}")

print()

print("Checking installed G-MASS CLI...")
scripts_dir = sysconfig.get_path("scripts")
gmass_exe = "gmass.exe" if os.name == "nt" else "gmass"
gmass_script = os.path.join(scripts_dir, gmass_exe) if scripts_dir else ""
gmass_cmd = (gmass_script if gmass_script and os.path.exists(gmass_script) else None) or shutil.which("gmass")
if gmass_cmd:
    result = subprocess.run(
        [gmass_cmd, "--help"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    if result.returncode == 0:
        print("  OK gmass CLI found")
    else:
        print("  WARN gmass CLI file found, but its launcher did not run")
        print("    Fallback: python -m run_bilingual_eval --help")
        print("    If this virtualenv was copied or moved, recreate it and rerun setup.sh.")
else:
    print("  FAIL gmass CLI not found - Run: python -m pip install -e .")
    errors.append("gmass CLI missing")
print()

print("Checking .env file...")
if not os.path.exists(".env"):
    print("  FAIL .env file not found in current directory")
    if os.path.exists(".env.example"):
        print("    Run setup.sh to generate a local .env from .env.example, then fill in your API keys.\n")
    else:
        print("    Create .env from your own environment or add .env.example to the repo.\n")
    errors.append(".env file missing")
else:
    if module_available("dotenv"):
        from dotenv import load_dotenv

        load_dotenv()
    else:
        print("  FAIL python-dotenv is not installed")
        errors.append("Missing package: dotenv")

    keys = {
        "HF_TOKEN": "huggingface.co -> Settings -> Access Tokens",
        "OPENAI_API_KEY": "platform.openai.com/api-keys",
        "GEMINI_API_KEY": "aistudio.google.com -> Get API Key",
    }

    for key, source in keys.items():
        value = os.getenv(key)
        if not value or "your_" in value.lower() or value.strip() == "" or value.lower() in {"changeme", "replace-me"}:
            print(f"  FAIL {key} - not set or is a placeholder")
            print(f"    Get it from: {source}")
            errors.append(f"Missing key: {key}")
        else:
            print(f"  OK {key} is set")

    print()

    if os.getenv("SCORER_BACKEND", "").lower() == "transformers" or any(
        os.getenv(key, "").lower() == "transformers"
        for key in ("PHI3_BACKEND", "BIOMISTRAL_BACKEND", "LOCAL_MODEL_BACKEND")
    ):
        print("Checking local Transformers packages...")
        local_packages = {
            "torch": "pip install -r requirements-local.txt",
            "transformers": "pip install -r requirements-local.txt",
            "accelerate": "pip install -r requirements-local.txt",
            "safetensors": "pip install -r requirements-local.txt",
            "sentencepiece": "pip install -r requirements-local.txt",
        }
        for pkg, install_cmd in local_packages.items():
            if module_available(pkg):
                print(f"  OK {pkg}")
            else:
                print(f"  FAIL {pkg} - Run: {install_cmd}")
                errors.append(f"Missing local package: {pkg}")
        print()

print("Checking folder structure...")
folders = ["models", "scorer", "probes", "outputs", "tests", "configs"]
for folder in folders:
    if os.path.isdir(folder):
        print(f"  OK {folder}/")
    else:
        os.makedirs(folder, exist_ok=True)
        print(f"  OK {folder}/ - created")

print()

print("=" * 60)
if not errors:
    print("  Environment ready. Run: python test_models.py")
else:
    print(f"  {len(errors)} issue(s) to fix before running tests:\n")
    for i, err in enumerate(errors, 1):
        print(f"    {i}. {err}")
    print("\n  Fix these, then run: python scripts/check_environment.py again")
print("=" * 60 + "\n")
