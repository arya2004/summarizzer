import textract
import re
import openpyxl
from openpyxl.utils.exceptions import IllegalCharacterError

# Extract text from the DOC file using textract
file_path = "sotr.doc"
text = textract.process(file_path).decode('utf-8')

# Print the first 2 lines of the extracted text
print("\n".join(text.split("\n")[:2]))


pattern = re.compile(r'\b([a-z])\) .+?(?=\n\s*\b[a-z]\) |\n\s*$)', re.DOTALL)
matches = pattern.finditer(text)

# Create a new Excel workbook and select the active worksheet
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Extracted Points"

# Write the matches to the Excel sheet
ws.append(["Point", "Content"])
for match in matches:
    try:
        ws.append([match.group(1), match.group(0).strip()])
    except IllegalCharacterError:
        continue

# Save the workbook
output_path = "extracted_points_1.xlsx"
wb.save(output_path)

print(f"Extracted points saved to {output_path}")
