from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

class PDFReportGenerator:
    """PDF 리포트 생성기"""
    
    def __init__(self, filename="report.pdf"):
        self.filename = filename
        # 한글 폰트 등록
        font_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",       # macOS
            "C:\\Windows\\Fonts\\malgun.ttf"                    # Windows
        ]
        
        # 시스템에 설치된 한글 폰트 찾기
        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break
                
        if font_path:
            try:
                pdfmetrics.registerFont(TTFont("KoreanFont", font_path))
                self.font_name = "KoreanFont"
            except Exception as e:
                print(f"한글 폰트 등록 실패: {e}")
                self.font_name = "Helvetica"
        else:
            print("한글 폰트를 찾을 수 없습니다.")
            self.font_name = "Helvetica"
    
    def generate_pdf(self, report_text):
        """리포트를 PDF로 저장
        
        Args:
            report_text (str): 리포트 텍스트
            
        Returns:
            str: 생성된 PDF 파일 경로
        """
        c = canvas.Canvas(self.filename, pagesize=letter)
        width, height = letter
        
        # 페이지 여백 설정
        margin = 80
        line_height = 14
        current_y = height - margin
        
        def add_page():
            nonlocal current_y
            c.showPage()
            current_y = height - margin
            # 새 페이지에 제목 추가
            c.setFont(self.font_name, 16)
            c.drawString(margin, height - margin, "부안군 정책 제안 리포트 (계속)")
            current_y -= 40
        
        # 첫 페이지 제목
        c.setFont(self.font_name, 16)
        c.drawString(margin, current_y, "부안군 정책 제안 리포트")
        current_y -= 40
        
        # 구분선
        c.setStrokeColorRGB(0.8, 0.8, 0.8)
        c.line(margin, current_y, width - margin, current_y)
        current_y -= 20
        
        # 본문
        c.setFont(self.font_name, 12)
        
        # 텍스트 줄바꿈 처리
        lines = report_text.split('\n')
        for line in lines:
            # 긴 줄은 자동 줄바꿈
            while len(line) > 0:
                if current_y < margin + line_height:  # 페이지 끝 체크
                    add_page()
                
                if len(line) <= 50:  # 한 줄 최대 길이
                    c.drawString(margin, current_y, line)
                    current_y -= line_height
                    break
                else:
                    c.drawString(margin, current_y, line[:50])
                    current_y -= line_height
                    line = line[80:]
            
            # 단락 구분을 위한 빈 줄
            if current_y < margin + line_height:
                add_page()
            current_y -= line_height
        
        c.save()
        return self.filename 