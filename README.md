# Sentiment Analysis Dashboard

A sentiment analysis dashboard built with Plotly Dash

## Prerequisites

- Python 3.14
- Git

## Setup

Choose one of the following setup methods:

### Option 1: Using uv (Recommended - Faster)

uv is a fast Python package manager that handles dependency resolution better than pip.

**First-time setup:**

```bash
# 1. Install uv (one-time)
pip install uv

# 2. Clone the repository
git clone https://github.com/Kancil-Capital/sentiment-analysis.git
cd sentiment-analysis

# 3. Create virtual environment and install dependencies
uv sync

# 4. Install the project as an editable package
uv pip install -e .

#5. Install the spaCy model needed (should take around 6 minutes)
uv run python -m spacy download en_core_web_lg
```

**Running the application:**

```bash
# Option A: Using uv run (no activation needed)
uv run python app/main.py

# Option B: Activate venv first, then run
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

python app/main.py
```

### Option 2: Using pip (Traditional)

**First-time setup:**

```bash
# 1. Clone the repository
git clone https://github.com/Kancil-Capital/sentiment-analysis.git
cd sentiment-analysis

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install the project as an editable package
pip install -e .
```

**Running the application:**

```bash
# Make sure venv is activated first
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# Run the app
python app/main.py
```

The dashboard will be available at: `http://localhost:8050`

## Adding New Dependencies

### With uv

```bash
# Add a package
uv add package-name

# Update requirements.txt and commit these files
uv pip compile pyproject.toml -o requirements.txt
git add pyproject.toml uv.lock
git commit -m "Add package-name"
```

### With pip

```bash
# Add a package
pip install package-name
```

Edit [pyproject.toml](pyproject.toml) and add the new package with version in dependencies list
Add the package and version in [requirements.txt](requirements.txt)

## Troubleshooting

### Import errors (`ModuleNotFoundError: No module named 'app'`)

Make sure you've installed the project as a package:

```bash
# With uv
uv pip install -e .

# With pip (venv must be activated)
pip install -e .
```

Or run app as a module

```bash
# Inside venv or with uv run
python -m app.main
```

### Dependencies out of sync after pulling

```bash
# With uv
uv sync

# With pip
source .venv/bin/activate
pip install -r requirements.txt
```

### Virtual environment issues

```bash
# Delete and recreate
rm -rf .venv/

# With uv
uv sync
uv pip install -e .

# With pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
