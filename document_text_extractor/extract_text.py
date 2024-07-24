import textract

def extract_text_from_file(file_path):
    """
    Extracts text from the specified file using textract.

    Args:
        file_path (str): The path to the file from which to extract text.

    Returns:
        str: The extracted text.
    """
    try:
        text = textract.process(file_path).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to extract text: {e}")
    return text
