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
    subpoints = {}
    current_main_point = None

    # Regular expressions for main points and subpoints
    main_point_pattern = re.compile(r'^\d+\.\d+')
    subpoint_pattern = re.compile(r'^\s*[a-z]\)')

    for line in text.split('\n'):
        line = line.strip()
        if main_point_pattern.match(line):
            current_main_point = line.split(' ')[0]
            subpoints[current_main_point] = []
        elif subpoint_pattern.match(line):
            if current_main_point:
                subpoints[current_main_point].append(line)
    
    return subpoints

def store_in_excel(subpoints, output_file):
    wb = Workbook()
    ws = wb.active
    ws.title = "Subpoints"

    ws.cell(row=1, column=1, value="Main Point")
    ws.cell(row=1, column=2, value="Subpoints")

    row = 2
    for main_point, subpoint_list in subpoints.items():
        ws.cell(row=row, column=1, value=main_point)
        ws.cell(row=row, column=2, value='\n'.join(subpoint_list))
        row += 1

    wb.save(output_file)

# Path to your DOC file
file_path = 'sotr.doc'
output_file = 'extracted_subpoints_v1.xlsx'

# Extract text
try:
    extracted_text = extract_text_from_doc(file_path)
    subpoints = parse_subpoints(extracted_text)
    store_in_excel(subpoints, output_file)
    print(f"Subpoints have been extracted and stored in {output_file}")
except Exception as e:
    print(e)
