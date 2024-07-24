import re

def find_text_points(text):
    """
    Finds points in the text that match the specified pattern.

    Args:
        text (str): The text to search through.

    Returns:
        list: A list of match objects.
    """
    pattern = re.compile(r'\b([a-z])\) .+?(?=\n\s*\b[a-z]\) |\n\s*$)', re.DOTALL)
    matches = pattern.finditer(text)
    return matches

def clean_text(text):
    """
    Cleans the text by removing excess whitespace, newlines, and tabs.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    cleaned_text = text.strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple whitespaces with a single space
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)  # Replace multiple newlines with a single newline
    cleaned_text = re.sub(r'\t+', '\t', cleaned_text)  # Remove any tabs
    return cleaned_text
