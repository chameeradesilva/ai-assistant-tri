"""Script to generate a test PDF with text and tables."""
from fpdf import FPDF
import os

def create_test_pdf():
    """Create a test PDF with text and a simple table."""
    pdf = FPDF()
    
    # Add a page
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", size=16)
    
    # Add a title
    pdf.cell(200, 10, txt="Test Document", ln=1, align='C')
    
    # Add some text
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="This is a test document with some text and a table.", ln=1, align='L')
    pdf.ln(10)
    
    # Add a table
    col_width = 40
    row_height = 10
    
    # Table headers
    headers = ["ID", "Name", "Age", "City"]
    for header in headers:
        pdf.cell(col_width, row_height, header, 1)
    pdf.ln(row_height)
    
    # Table data
    data = [
        ["1", "John", "25", "New York"],
        ["2", "Alice", "30", "London"],
        ["3", "Bob", "28", "Paris"],
        ["4", "Carol", "35", "Tokyo"]
    ]
    
    for row in data:
        for item in row:
            pdf.cell(col_width, row_height, item, 1)
        pdf.ln(row_height)
    
    # Save the PDF
    os.makedirs("data/pdfs", exist_ok=True)
    pdf_path = "data/pdfs/TEST01.pdf"
    pdf.output(pdf_path)
    print(f"Created test PDF at: {pdf_path}")

if __name__ == "__main__":
    create_test_pdf() 