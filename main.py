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
    subpoint_pattern = re.compile(r'^\s*[a-z]\)|^\s*\(\w\)|^\s*\d+\)|^\s*i{1,3}\)', re.MULTILINE)

    for match in subpoint_pattern.finditer(text):
        subpoint_text = match.group().strip()
        start = match.end()
        end = text.find('\n', start)
        if end == -1:
            end = len(text)
        subpoint_content = text[start:end].strip()
        subpoints.append(f"{subpoint_text} {subpoint_content}")
    
    return subpoints

def store_in_excel(subpoints, output_file):
    wb = Workbook()
    ws = wb.active
    ws.title = "Subpoints"

    ws.cell(row=1, column=1, value="Subpoints")

    for row, subpoint in enumerate(subpoints, start=2):
        ws.cell(row=row, column=1, value=subpoint)

    wb.save(output_file)

# Path to your DOC file
file_path = 'sotr.doc'
output_file = 'extracted_subpoints_v2.xlsx'

# Extract text
try:
    extracted_text = extract_text_from_doc(file_path)
    subpoints = parse_subpoints(extracted_text)
    store_in_excel(subpoints, output_file)
    print(f"Subpoints have been extracted and stored in {output_file}")
except Exception as e:
    print(e)
