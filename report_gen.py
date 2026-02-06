from fpdf import FPDF
from datetime import datetime
import re

class ForensicReport(FPDF):
    def header(self):
        # Dark header bar
        self.set_fill_color(10, 11, 13) 
        self.rect(0, 0, 210, 40, 'F')
        self.set_font('Courier', 'B', 20)
        self.set_text_color(0, 242, 255) # Cyan
        self.cell(0, 20, 'TACTICAL FORENSIC REPORT', 0, 1, 'C')
        self.ln(10)

def clean_text(text):
    """
    Removes Emojis and non-latin-1 characters to prevent UnicodeEncodeError.
    Essential for Streamlit Cloud stability.
    """
    if not text:
        return ""
    # Remove Emojis and special symbols
    text = re.sub(r'[^\x00-\x7F]+', ' ', str(text)) 
    # Ensure it's strictly latin-1 compatible
    return text.encode('latin-1', 'ignore').decode('latin-1')

def create_pdf_report(results_list, case_notes=""):
    pdf = ForensicReport()
    pdf.add_page()
    pdf.set_font("Courier", size=11)
    
    # Header Info
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, txt=f"DATE: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.ln(5)

    # Conclusion Section
    pdf.set_font("Courier", 'B', 14)
    pdf.set_text_color(139, 0, 0) # Dark Red for emphasis
    pdf.cell(0, 10, txt="INVESTIGATOR CONCLUSION:", ln=True)
    pdf.set_font("Courier", size=10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, txt=clean_text(case_notes))
    pdf.ln(10)

    # Exhibits
    for res in results_list:
        pdf.set_draw_color(0, 242, 255) # Cyan separator
        pdf.cell(0, 0, '', 'T', ln=True)
        pdf.ln(5)
        
        pdf.set_font("Courier", 'B', 12)
        pdf.cell(0, 8, txt=f"EXHIBIT: {clean_text(res['FILENAME'])}", ln=True)
        
        pdf.set_font("Courier", size=10)
        # Clean the verdict to remove the flag emoji for the PDF
        verdict_only = res['VERDICT'].replace("🚩 ", "").replace("🏳️ ", "")
        
        pdf.cell(0, 6, txt=f"VERDICT: {verdict_only} | CONFIDENCE: {res['CONFIDENCE']:.2f}%", ln=True)
        pdf.multi_cell(0, 6, txt=f"METADATA: {clean_text(res['METADATA'])}")
        pdf.ln(5)

    # Return the bytes
    return pdf.output(dest='S').encode('latin-1')