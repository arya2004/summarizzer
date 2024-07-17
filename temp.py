import textract
import re
import openpyxl
from openpyxl.utils.exceptions import IllegalCharacterError

file_path = "sotr.doc"
text = textract.process(file_path).decode('utf-8')




point_pattern = re.compile(r'\b([a-z])\) .+?(?=\n\s*\b[a-z]\) |\n\s*$)', re.DOTALL)
points = point_pattern.finditer(text)


wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Extracted Points"

ws.append(["Point", "Content"])


for match in points:
    try:
        point_text = match.group(0).strip()
        if '|' in point_text:
            continue  
        cleaned_text = re.sub(r'\s+', ' ', point_text) 
        cleaned_text = re.sub(r'\n+', '\n', cleaned_text)  
        cleaned_text = re.sub(r'\t+', '\t', cleaned_text)  
        ws.append([match.group(1), cleaned_text])
    except IllegalCharacterError:
        continue


output_path = "extracted_points_31.xlsx"
wb.save(output_path)

print(f"Extracted points saved to {output_path}")
