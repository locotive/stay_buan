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
        self.client = openai.OpenAI(api_key=api_key)
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
            
        required_columns = ['platform', 'sentiment', 'content']
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
        
         # 실제 게시글처럼 보이도록 content 컬럼에서 샘플링
         sample_contents = df['content'].dropna().sample(min(10, len(df))).unique()
         sample_text = '\n'.join([f"- {content[:100]}..." for content in sample_contents])
        
         # 프롬프트 생성
         prompt = f"""다음 데이터를 바탕으로 부안군의 정책 제안 리포트를 작성해주세요.
         # 프롬프트 생성 (관광·정주 분석 섹션 추가)
         prompt = f"부안군 감성 분석 및 산업·정주 여건 결과를 바탕으로 정책 제안 리포트를 작성해주세요.

 [관광 산업 분석 결과]
 - 연간 관광객 수: 약 700만명 (2019년) → 500만명 (2023년, –37%)
 - 고령화율: 전북 평균 24% 대비 높음
 - 주요 관광지: 변산반도 · 채석강 · 내소사
   (문제) 편의시설·교통 불편으로 만족도 저하

 [정주 여건 분석 결과]
 - 의료 서비스 부족: 응급 접근성 낮음
 - 교육 환경 열악: 낡은 학교·교육기관 부족
 - 정책 지원 미흡: 현행 법령 대비 예산·인력 부족
 - 교통 인프라 미흡: 대중교통 연결성 약함

 [데이터 요약]
 {summary_text}

 [샘플 게시글]
 {sample_text}

 [지시사항]
0. 답변 시작 부분에 "분석된 데이터 요약 결과에 따르면 …" 와 같이, 위 [관광 산업 분석 결과]와 [정주 여건 분석 결과]에서 보이는 주요 수치와 특징을 언급해주세요. 추가로 - 시계열 감성 트렌드에서는 긍정 비율이 점차 감소하고 부정 비율이 상승하는 추세가 관측됩니다.  이런 내용도 넣어줘

 1. 다음 형식으로 리포트를 작성해주세요:
    - 현황 분석
    - 문제점 도출
    - 정책 제안 (3-5개)
    - 기대효과

 2. 정책 제안 시 다음을 반영해주세요:
    - 관광 산업 활성화 (인프라 개선, 노년층 맞춤형 프로그램)
    - 정주 여건 개선 (의료·교육 서비스 확충, 교통망 강화)

 3. 각 정책에는:
    - 정책명
    - 추진 방향
    - 필요 예산 (예상)
    - 추진 일정(최소 현시점(2025하반기) 이후로 시작 시점을 지정해야함)

 4. 구체적인 수치와 근거를 포함하고, 부안군 특성에 맞는 실현 가능한 방안을 제안해주세요.
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
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo-16k",  # 더 긴 컨텍스트를 지원하는 모델 사용
                    messages=[
                        {"role": "system", "content": "당신은 부안군의 정책 제안 전문가입니다. 데이터에 기반한 구체적이고 실현 가능한 정책을 제안해주세요. 각 정책에 대해 상세한 설명과 구체적인 실행 방안을 포함해주세요."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4000,  # 토큰 제한 증가
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