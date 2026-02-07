from fpdf import FPDF
from datetime import datetime
import re

class ForensicReport(FPDF):
    def header(self):
        # Professional Tactical Header
        self.set_fill_color(10, 11, 13) 
        self.rect(0, 0, 210, 40, 'F')
        self.set_font('Courier', 'B', 18)
        self.set_text_color(0, 242, 255) 
        self.cell(0, 20, 'ForensiX-Image Forgery Detector', 0, 1, 'C')
        self.ln(10)

def clean_text(text):
    """Ensures text is compatible with PDF encoding by stripping non-standard characters."""
    if not text: return ""
    text = re.sub(r'[^\x20-\x7E]+', ' ', str(text)) 
    return text.encode('latin-1', 'ignore').decode('latin-1')

def create_pdf_report(results_list, case_notes=""):
    pdf = ForensicReport()
    pdf.add_page()
    pdf.set_font("Courier", size=11)
    pdf.cell(0, 10, txt=f"DATE: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Courier", 'B', 14)
    pdf.set_text_color(139, 0, 0)
    pdf.cell(0, 10, txt="OFFICIAL INVESTIGATOR CONCLUSION:", ln=True)
    pdf.set_font("Courier", size=10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, txt=clean_text(case_notes))
    pdf.ln(10)

    for res in results_list:
        pdf.set_draw_color(0, 242, 255)
        pdf.cell(0, 0, '', 'T', ln=True)
        pdf.ln(5)
        pdf.set_font("Courier", 'B', 12)
        pdf.cell(0, 8, txt=f"EXHIBIT: {clean_text(res['FILENAME'])}", ln=True)
        # Remove Emojis for PDF stability
        v_clean = res['VERDICT'].replace("🚩 ", "").replace("🏳️ ", "")
        pdf.set_font("Courier", size=10)
        pdf.cell(0, 6, txt=f"VERDICT: {v_clean} | AI CONFIDENCE: {res['CONFIDENCE']:.2f}%", ln=True)
        pdf.multi_cell(0, 6, txt=f"FORENSIC TRACE: {clean_text(res['METADATA'])}")
        pdf.ln(5)

    return pdf.output(dest='S').encode('latin-1')