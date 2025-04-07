import pandas as pd
from openai import GPT

class ReportGenerator:
    """정책 제안 및 리포트 생성기"""
    
    def __init__(self, api_key):
        self.gpt = GPT(api_key)
    
    def generate_summary(self, data):
        """데이터 요약 및 정책 제안 생성"""
        # 데이터 요약
        summary = data.describe()
        
        # GPT를 통한 정책 제안
        prompt = f"다음 데이터를 바탕으로 부안군의 정책 제안을 작성해 주세요: {summary}"
        response = self.gpt.generate(prompt)
        
        return response

    def save_report(self, summary, filename="report.pdf"):
        """리포트 저장"""
        with open(filename, 'w') as f:
            f.write(summary) 