import openai
import pandas as pd

class GPTReportGenerator:
    """GPT 기반 리포트 생성기"""
    
    def __init__(self, api_key):
        openai.api_key = api_key
    
    def generate_report(self, data):
        """데이터 요약을 바탕으로 리포트 생성"""
        df = pd.DataFrame(data)
        
        # 1. 컬럼 존재 여부 확인
        required_columns = ['platform', 'sentiment', 'keyword']
        for col in required_columns:
            if col not in df.columns:
                print(f"⚠️ '{col}' 컬럼이 누락되어 기본값으로 채웁니다.")
                if col == 'sentiment':
                    df[col] = 'unknown'
                else:
                    df[col] = 'unknown'
        
        if df.empty:
            return "⚠️ 리포트를 생성할 수 없습니다: 데이터가 비어 있습니다."

        # 2. 플랫폼-감정별 집계
        platform_summary = df.groupby(['platform', 'sentiment']).size().unstack().fillna(0)
        summary_text = platform_summary.to_string()
        
        # 3. 프롬프트 생성
        keywords = ', '.join(df['keyword'].dropna().unique())
        prompt = f"다음 데이터 요약을 바탕으로 부안군의 정책 제안을 작성해 주세요:\n{summary_text}\n주요 키워드: {keywords}"
        
        # 4. GPT 호출
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "당신은 정책 제안 전문가입니다."},
                      {"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
