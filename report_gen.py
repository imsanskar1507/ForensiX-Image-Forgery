from fpdf import FPDF
from datetime import datetime

class ForensicReport(FPDF):
    def header(self):
        self.set_fill_color(20, 20, 25)
        self.rect(0, 0, 210, 40, 'F')
        self.set_font('Courier', 'B', 18)
        self.set_text_color(139, 0, 0)
        self.cell(0, 20, 'CLASSIFIED FORENSIC REPORT', 0, 1, 'C')
        self.ln(5)

def clean_text(text):
    return str(text).encode('latin-1', 'ignore').decode('latin-1')

def create_pdf_report(results_list, case_notes="No agent notes."):
    pdf = ForensicReport()
    pdf.add_page()
    pdf.set_font("Courier", size=12)

    # Agent Details
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, txt=f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Courier", 'B', 14)
    pdf.set_text_color(139, 0, 0)
    pdf.cell(0, 10, txt="AGENT CONCLUSIONS:", ln=True)
    pdf.set_font("Courier", size=11)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 7, txt=clean_text(case_notes))
    pdf.ln(10)

    for res in results_list:
        pdf.set_draw_color(139, 0, 0)
        pdf.set_line_width(0.5)
        pdf.cell(0, 0, '', 'T', ln=True)
        pdf.ln(5)
        
        pdf.set_font("Courier", 'B', 12)
        pdf.cell(0, 10, txt=f"EXHIBIT: {clean_text(res['FILENAME'])}", ln=True)
        pdf.set_font("Courier", size=10)
        pdf.cell(0, 8, txt=f"VERDICT: {clean_text(res['VERDICT'])} | AI CONFIDENCE: {res['CONFIDENCE']:.2f}%", ln=True)
        pdf.multi_cell(0, 6, txt=f"TRACE: {clean_text(res['METADATA'])}")
        pdf.ln(5)

    return pdf.output(dest='S').encode('latin-1')