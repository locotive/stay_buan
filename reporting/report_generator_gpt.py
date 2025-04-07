import openai
import pandas as pd

class GPTReportGenerator:
    """GPT 기반 리포트 생성기"""
    
    def __init__(self, api_key):
        openai.api_key = api_key
    
    def generate_report(self, data):
        """데이터 요약을 바탕으로 리포트 생성"""
        # 플랫폼별, 감정별 요약
        platform_summary = data.groupby(['platform', 'sentiment']).size().unstack().fillna(0)
        summary_text = platform_summary.to_string()
        
        prompt = f"다음 데이터 요약을 바탕으로 부안군의 정책 제안을 작성해 주세요:\n{summary_text}\n주요 키워드: {', '.join(data['keyword'].unique())}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=500
        )
        return response.choices[0].text.strip() 