# Fixing NLTK Resource Issues

This guide explains how to fix NLTK (Natural Language Toolkit) resource-related issues in the IridoVeda application.

## Common Error Messages

If you encounter errors like:

```
LookupError: Resource punkt not found.
Please use the NLTK Downloader to obtain the resource: 
>>> import nltk 
>>> nltk.download('punkt')
```

This means that NLTK is missing required resources for text processing.

## Automatic Fixes

We've included several tools to automatically fix NLTK resource issues:

### 1. Using the Fix Script

Run the provided fix script:

```bash
# For local installations
./fix_nltk_resources.py

# For Docker installations
./fix_nltk_docker.sh
```

### 2. Manual Download in Python

You can manually download the NLTK resources by opening a Python shell and running:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### 3. Rebuilding Docker Container

If you're using Docker and still experiencing issues:

```bash
# Stop the existing containers
docker-compose down

# Rebuild without using cache
docker-compose up --build --no-cache
```

## Verifying the Fix

To verify that NLTK resources are properly installed:

```python
import nltk
from nltk.tokenize import sent_tokenize

# Test sentence tokenization
print(sent_tokenize("This is a test. This is another test."))
```

You should see output like:
```
['This is a test.', 'This is another test.']
```

## Common Issues and Solutions

### Permission Issues

If you encounter permission errors when downloading NLTK resources:

```bash
# Run with sudo (Linux/MacOS)
sudo python -c "import nltk; nltk.download('punkt')"

# Or specify a custom download directory
python -c "import nltk; nltk.download('punkt', download_dir='./nltk_data')"
```

### Docker Volume Persistence

If you're using Docker with volumes, ensure that the NLTK data directory is properly mounted:

```yaml
# In docker-compose.yml
volumes:
  - ./nltk_data:/root/nltk_data
```

### Environment Variables

You can specify the NLTK data directory using environment variables:

```bash
export NLTK_DATA=/path/to/nltk_data
```

For Docker, add this to your docker-compose.yml:

```yaml
environment:
  - NLTK_DATA=/app/nltk_data
```

## Additional Resources

- [NLTK Data Installation Guide](https://www.nltk.org/data.html)
- [Docker and NLTK Best Practices](https://www.nltk.org/install.html)
