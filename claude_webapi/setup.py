from setuptools import setup, find_packages

setup(
    name="claude-webapi",
    version="1.0.0",
    description="Reverse-engineered asynchronous Python wrapper for the Claude.ai web app",
    long_description=open("../README.md", encoding="utf-8").read()
    if __import__("os").path.exists("../README.md")
    else "",
    long_description_content_type="text/markdown",
    author="Wojciech Dudek",
    author_email="wojtek.dudek.pl@gmail.com",
    url="https://github.com/your-username/Claude-API",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(where=".."),
    package_dir={"": ".."},
    install_requires=[
        "aiohttp>=3.9",
    ],
    extras_require={
        "browser": ["browser-cookie3>=0.19"],
        "dev": ["pytest>=7", "pytest-asyncio>=0.23", "black", "ruff"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Communications :: Chat",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="claude anthropic ai chatbot async web-api reverse-engineered",
)
