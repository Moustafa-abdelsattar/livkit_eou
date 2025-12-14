"""
Setup configuration for livekit-plugins-arabic-turn-detector
"""

from setuptools import setup, find_packages
import os

# Read version from version.py
version_file = os.path.join(
    os.path.dirname(__file__),
    "livekit_plugins_arabic_turn_detector",
    "version.py"
)
with open(version_file) as f:
    exec(f.read())

# Read README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="livekit-plugins-arabic-turn-detector",
    version=__version__,  # noqa: F821
    author="Moustafa3092",
    description="Fine-tuned Arabic End-of-Utterance detection for LiveKit voice agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Moustafa3092/livekit-plugins-arabic-turn-detector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "livekit-agents>=1.3.0",
        "livekit-plugins-turn-detector>=1.3.0",
        "transformers>=4.45.0",
        "torch>=2.0.0",
        "huggingface-hub>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
)
