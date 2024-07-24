import os
from extract_text import extract_text_from_file
from text_processing import find_text_points, clean_text
from write_to_excel import write_points_to_excel


def main():
    # Directory paths
    data_dir = "../data"
    output_dir = "../output"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # File path to the document
    file_path = os.path.join(data_dir, "sotr.doc")
    
    # Extract text from the file
    text = extract_text_from_file(file_path)
    
    # Find and clean text points
    matches = find_text_points(text)
    points = [(match.group(1), clean_text(match.group(0))) for match in matches]
    
    # Output file path
    output_path = os.path.join(output_dir, "extracted_points_3.xlsx")
    
    # Write points to an Excel file
    write_points_to_excel(points, output_path)

if __name__ == "__main__":
    main()
