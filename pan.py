import textract
import re
import openpyxl
from openpyxl.utils.exceptions import IllegalCharacterError

file_path = "sotr.doc"
text = textract.process(file_path).decode('utf-8')

print(text[15500:19000])

# Pattern to match points (a) ... (z) and potential tables
point_pattern = re.compile(r'\b([a-z])\) .+?(?=\n\s*\b[a-z]\) |\n\s*$)', re.DOTALL)
table_pattern = re.compile(r'(?:\n\s*[-\w]+\s*[-\w]+)+\n', re.DOTALL)  # Adjust this pattern based on table structure

points = point_pattern.finditer(text)
tables = table_pattern.finditer(text)

# Create workbook
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Extracted Points"

ws.append(["Point", "Content"])

# Extract tables and replace them with a placeholder in the text
table_dict = {}
for i, table in enumerate(tables):
    table_dict[f"TABLE_PLACEHOLDER_{i}"] = table.group(0)
    text = text.replace(table.group(0), f"TABLE_PLACEHOLDER_{i}")

# Process points and reinsert tables
for match in points:
    try:
        cleaned_text = match.group(0).strip()
        for placeholder, table in table_dict.items():
            cleaned_text = cleaned_text.replace(placeholder, table)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple whitespaces with a single space
        cleaned_text = re.sub(r'\n+', '\n', cleaned_text)  # Replace multiple newlines with a single newline
        cleaned_text = re.sub(r'\t+', '\t', cleaned_text)  # Replace any tabs with a single tab
        ws.append([match.group(1), cleaned_text])
    except IllegalCharacterError:
        continue

# Save the workbook
output_path = "extracted_points_12.xlsx"
wb.save(output_path)

print(f"Extracted points saved to {output_path}")
