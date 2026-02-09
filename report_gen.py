from fpdf import FPDF
from datetime import datetime
import re

class ForensicReport(FPDF):
    def header(self):
        self.set_fill_color(10, 11, 13) 
        self.rect(0, 0, 210, 40, 'F')
        self.set_font('Courier', 'B', 18)
        self.set_text_color(0, 242, 255) 
        self.cell(0, 20, 'ForensiX-Image Forgery Detector', 0, 1, 'C')
        self.ln(10)

def create_pdf_report(results_list, case_notes=""):
    pdf = ForensicReport()
    pdf.add_page()
    pdf.set_font("Courier", size=11)
    pdf.cell(0, 10, txt=f"GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, txt=f"LOCATION: NAGPUR_MS_IN", ln=True)
    pdf.ln(10)
    # ... rest of report logic ...
    return pdf.output(dest='S').encode('latin-1')