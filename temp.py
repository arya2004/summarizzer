import textract
import re
from openpyxl import Workbook

def extract_text_from_doc(file_path):
    try:
        text = textract.process(file_path)
        return text.decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing the DOC file: {e}")

def parse_subpoints(text):
    subpoints = []
    main_point_pattern = re.compile(r'^\d+\.\d+')
    subpoint_pattern = re.compile(r'^\s*[a-z]\)')
    nested_subpoint_pattern = re.compile(r'^\s*[ivxlc]+\)')

    lines = text.split('\n')
    current_main_point = None
    current_subpoint = None

    for line in lines:
        line = line.strip()
        if main_point_pattern.match(line):
            current_main_point = line
            current_subpoint = None
        elif subpoint_pattern.match(line):
            current_subpoint = line
            subpoints.append((current_main_point, current_subpoint, None, line))
        elif nested_subpoint_pattern.match(line):
            if current_subpoint:
                subpoints.append((current_main_point, current_subpoint, line, None))
            else:
                subpoints.append((current_main_point, None, line, None))
    
    return subpoints

def store_in_excel(subpoints, output_file):
    wb = Workbook()
    ws = wb.active
    ws.title = "Subpoints"

    ws.cell(row=1, column=1, value="Main Point")
    ws.cell(row=1, column=2, value="Subpoint")
    ws.cell(row=1, column=3, value="Nested Subpoint")

    row = 2
    for main_point, subpoint, nested_subpoint, full_text in subpoints:
        ws.cell(row=row, column=1, value=main_point)
        ws.cell(row=row, column=2, value=subpoint)
        ws.cell(row=row, column=3, value=nested_subpoint)
        row += 1

    wb.save(output_file)

# Path to your DOC file
file_path = 'sotr.doc'
output_file = 'extracted_subpoints_v4.xlsx'

# Extract text
try:
    extracted_text = extract_text_from_doc(file_path)
    subpoints = parse_subpoints(extracted_text)
    store_in_excel(subpoints, output_file)
    print(f"Subpoints have been extracted and stored in {output_file}")
except Exception as e:
    print(e)
