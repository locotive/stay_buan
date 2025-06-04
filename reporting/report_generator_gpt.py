import openai
import pandas as pd
import os
from typing import Dict, List, Optional
import logging

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class GPTReportGenerator:
    """GPT 기반 리포트 생성기"""

    def __init__(self, api_key: str):
        """GPT 리포트 생성기 초기화
        
        Args:
            api_key (str): OpenAI API 키
        """
        import openai
        openai.api_key = api_key
        # API 키 길이 및 일부 마스킹해서 로깅
        masked = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]
        logger.debug(f"OpenAI API key configured: {masked}")

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """데이터 유효성 검사
        
        Args:
            df (pd.DataFrame): 검사할 데이터프레임
            
        Returns:
            bool: 데이터가 유효한지 여부
        """
        if df.empty:
            print("⚠️ 데이터가 비어 있습니다.")
            return False
            
        required_columns = ['platform', 'sentiment', 'keyword']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ 누락된 컬럼: {', '.join(missing_columns)}")
            for col in missing_columns:
                df[col] = 'unknown'
        
        return True

    def _create_prompt(self, df: pd.DataFrame) -> str:
        """프롬프트 생성
        
        Args:
            df (pd.DataFrame): 분석 데이터
            
        Returns:
            str: 생성된 프롬프트
        """
        # 플랫폼-감정별 집계
        platform_summary = df.groupby(['platform', 'sentiment']).size().unstack().fillna(0)
        summary_text = platform_summary.to_string()
        
        # 키워드 추출
        keywords = ', '.join(df['keyword'].dropna().unique())
        
        # 프롬프트 생성
        prompt = f"""다음 데이터를 바탕으로 부안군의 정책 제안 리포트를 작성해주세요.

[데이터 요약]
{summary_text}

[주요 키워드]
{keywords}

[지시사항]
1. 다음 형식으로 리포트를 작성해주세요:
   - 현황 분석
   - 문제점 도출
   - 정책 제안 (3-5개)
   - 기대효과

2. 각 정책 제안은 다음 항목을 포함해주세요:
   - 정책명
   - 추진 방향
   - 필요 예산 (예상)
   - 추진 일정

3. 데이터에 기반한 구체적인 수치와 근거를 포함해주세요.

4. 부안군의 특성을 고려한 실현 가능한 정책을 제안해주세요.
"""
        return prompt

    def generate_report(self, data: pd.DataFrame) -> str:
        """데이터 요약을 바탕으로 리포트 생성
        
        Args:
            data (pd.DataFrame): 분석 데이터
            
        Returns:
            str: 생성된 리포트
        """
        try:
            # 데이터프레임 변환
            df = pd.DataFrame(data)
            
            # 데이터 유효성 검사
            if not self._validate_data(df):
                return "⚠️ 리포트를 생성할 수 없습니다: 데이터가 유효하지 않습니다."
            
            # 프롬프트 생성
            prompt = self._create_prompt(df)
            logger.debug(f"Generated prompt: {prompt[:200]}...")  # 처음 200자만 로깅
            
            # GPT 호출
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "당신은 부안군의 정책 제안 전문가입니다. 데이터에 기반한 구체적이고 실현 가능한 정책을 제안해주세요."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
                
            except Exception as api_error:
                error_msg = str(api_error)
                logger.error(f"OpenAI API 호출 오류: {error_msg}", exc_info=True)
                if "rate_limit" in error_msg.lower():
                    return "⚠️ API 호출 한도가 초과되었습니다. 잠시 후 다시 시도해주세요."
                elif "invalid_request" in error_msg.lower():
                    return f"⚠️ 잘못된 API 요청: {error_msg}"
                else:
                    return f"⚠️ API 오류 발생: {error_msg}"
                
        except Exception as e:
            print(f"리포트 생성 중 오류 발생: {str(e)}")
            return f"⚠️ 리포트 생성 중 오류가 발생했습니다: {str(e)}" 