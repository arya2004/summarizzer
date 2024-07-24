import openpyxl
from openpyxl.utils.exceptions import IllegalCharacterError

def write_points_to_excel(points, output_path):
    """
    Writes the extracted points to an Excel file.

    Args:
        points (list): A list of tuples containing points and their content.
        output_path (str): The path to the output Excel file.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Extracted Points"
    ws.append(["Point", "Content"])

    for point, content in points:
        try:
            ws.append([point, content])
        except IllegalCharacterError:
            continue

    wb.save(output_path)
    print(f"Extracted points saved to {output_path}")
