import textract
import re
import openpyxl
from openpyxl.utils.exceptions import IllegalCharacterError

file_path = "sotr.doc"
text = textract.process(file_path).decode('utf-8')

print(text[15500:19000])

pattern = re.compile(r'\b([a-z])\) .+?(?=\n\s*\b[a-z]\) |\n\s*$)', re.DOTALL)
matches = pattern.finditer(text)

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Extracted Points"

ws.append(["Point", "Content"])
for match in matches:
    try:
        cleaned_text = match.group(0).strip()
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple whitespaces with a single space
        cleaned_text = re.sub(r'\n+', '\n', cleaned_text)  # Replace multiple newlines with a single newline
        cleaned_text = re.sub(r'\t+', '\t', cleaned_text)    # Remove any tabs
        ws.append([match.group(1), cleaned_text])
    except IllegalCharacterError:
        continue

# Save the workbook
output_path = "extracted_points_10.xlsx"
wb.save(output_path)

print(f"Extracted points saved to {output_path}")
