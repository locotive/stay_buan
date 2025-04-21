from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

class PDFReportGenerator:
    """PDF 리포트 생성기"""
    
    def __init__(self, filename="report.pdf"):
        self.filename = filename
    
    def generate_pdf(self, report_text):
        """리포트를 PDF로 저장"""
        c = canvas.Canvas(self.filename, pagesize=letter)
        width, height = letter
        
        c.drawString(100, height - 100, "부안군 정책 제안 리포트")
        text_object = c.beginText(100, height - 150)
        text_object.setFont("Helvetica", 12)
        text_object.textLines(report_text)
        
        c.drawText(text_object)
        c.save() 