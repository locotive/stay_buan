import os
import requests
import time
import random
import json
import logging
import hashlib
from urllib.parse import quote
from datetime import datetime

from core.base_crawler import BaseCrawler
from core.sentiment_analysis_ensemble import EnsembleSentimentAnalyzer

class GoogleSearchCrawler(BaseCrawler):
    """구글 검색 API를 이용한 크롤러"""

    def __init__(self, keywords, max_pages=10, save_dir="data/raw", analyze_sentiment=True, max_daily_queries=100):
        """
        구글 검색 크롤러 초기화
        
        Args:
            keywords: 검색할 키워드 리스트
            max_pages: 수집할 최대 페이지 수 (한 페이지당 10개 결과)
            save_dir: 저장 디렉토리
            analyze_sentiment: 감성 분석 수행 여부
            max_daily_queries: 일일 최대 API 쿼리 수 (기본값 100, 무료 사용량)
        """
        super().__init__(keywords, max_pages, save_dir)
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        if not self.api_key or not self.search_engine_id:
            self.logger.error("구글 API 키 또는 검색 엔진 ID가 없습니다. 환경변수 GOOGLE_API_KEY와 GOOGLE_SEARCH_ENGINE_ID를 설정해주세요.")
            raise ValueError("구글 API 키 또는 검색 엔진 ID가 없습니다.")
            
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.doc_ids = set()  # 중복 문서 확인용
        self.analyze_sentiment = analyze_sentiment
        self.max_daily_queries = max_daily_queries
        self.query_count = 0
        
        # 감성 분석기 초기화 (지연 로딩)
        self.sentiment_analyzer = None
        
        # 로깅 레벨 설정
        self.logger.setLevel(logging.INFO)
        
        # 원본 키워드 객체 저장
        if isinstance(keywords, list) and all(isinstance(k, dict) for k in keywords):
            self.original_keywords = keywords
            # 필터링용 키워드 문자열 추출
            self.keywords = [k['text'] for k in keywords]
        else:
            # 문자열 리스트인 경우 형식 변환
            self.keywords = keywords if isinstance(keywords, list) else [keywords]
            self.original_keywords = [{'text': k, 'condition': 'AND'} for k in self.keywords]
            
        # 일일 쿼리 카운트 파일 경로
        self.query_count_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'google_query_count.json')
        # 쿼리 카운트 로드
        self._load_query_count()

    def _load_query_count(self):
        """오늘의 API 쿼리 카운트 로드"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        try:
            if os.path.exists(self.query_count_file):
                with open(self.query_count_file, 'r') as f:
                    data = json.load(f)
                    
                # 오늘 날짜의 카운트가 있으면 로드, 아니면 0으로 초기화
                if data.get('date') == today:
                    self.query_count = data.get('count', 0)
                else:
                    # 날짜가 다르면 새로운 날이므로 카운트 초기화
                    self.query_count = 0
            else:
                # 파일이 없으면 초기화
                self.query_count = 0
                
            self.logger.info(f"오늘({today}) 사용된 Google API 쿼리 수: {self.query_count}/{self.max_daily_queries}")
        except Exception as e:
            self.logger.error(f"쿼리 카운트 로드 중 오류: {str(e)}")
            self.query_count = 0
    
    def _save_query_count(self):
        """현재 API 쿼리 카운트 저장"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # data 디렉토리 생성
            os.makedirs(os.path.dirname(self.query_count_file), exist_ok=True)
            
            # 카운트 저장
            with open(self.query_count_file, 'w') as f:
                json.dump({
                    'date': today,
                    'count': self.query_count
                }, f)
                
            self.logger.info(f"Google API 쿼리 카운트 업데이트: {self.query_count}/{self.max_daily_queries}")
        except Exception as e:
            self.logger.error(f"쿼리 카운트 저장 중 오류: {str(e)}")
    
    def _increment_query_count(self):
        """API 쿼리 카운트 증가"""
        self.query_count += 1
        self._save_query_count()
        
        # 쿼리 한도에 도달하면 경고
        if self.query_count >= self.max_daily_queries:
            self.logger.warning(f"⚠️ 일일 Google API 쿼리 한도({self.max_daily_queries}개)에 도달했습니다!")
            return False
        
        # 80% 이상 사용했을 때 경고
        if self.query_count >= self.max_daily_queries * 0.8:
            self.logger.warning(f"⚠️ Google API 쿼리 사용량이 {self.query_count}/{self.max_daily_queries}로 80% 이상 사용되었습니다.")
        
        return True

    def _get_sentiment_analyzer(self):
        """감성 분석기 지연 로딩"""
        if self.sentiment_analyzer is None:
            self.sentiment_analyzer = EnsembleSentimentAnalyzer()
        return self.sentiment_analyzer
        
    def analyze_text_sentiment(self, text):
        """텍스트 감성 분석"""
        if not self.analyze_sentiment or not text:
            return None, None
            
        try:
            analyzer = self._get_sentiment_analyzer()
            sentiment, confidence = analyzer.predict(text)
            return sentiment, confidence
        except Exception as e:
            self.logger.error(f"감성 분석 중 오류: {str(e)}")
            return None, None

    def _clean_html_content(self, text):
        """HTML 태그를 제거하고 텍스트 정리"""
        import re
        # HTML 태그 제거
        clean_text = re.sub(r'<[^>]*>', '', text)
        # 중복 공백 제거
        clean_text = re.sub(r'\s+', ' ', clean_text)
        # 특수 문자 처리
        clean_text = clean_text.replace('&nbsp;', ' ')
        clean_text = clean_text.replace('&lt;', '<')
        clean_text = clean_text.replace('&gt;', '>')
        clean_text = clean_text.replace('&amp;', '&')
        clean_text = clean_text.replace('&quot;', '"')
        
        return clean_text.strip()

    def _postprocess(self, items, original_keywords):
        """
        수집된 아이템을 후처리하는 메서드
        
        1. AND/OR 조건에 따라 키워드 필터링
        2. URL+제목 기반 해시로 중복 제거
        """
        processed_items = []
        filtered_count = 0
        
        for item in items:
            # 1. 키워드 조건 확인
            combined_text = f"{item['title']} {item['content']}"
            if not self._check_keyword_conditions(combined_text, original_keywords):
                self.logger.debug(f"키워드 필터링: 조건에 맞지 않음 - {item['title'][:30]}...")
                filtered_count += 1
                continue
                
            # 2. 중복 문서 확인
            doc_id = hashlib.sha256((item['url'] + item['title']).encode()).hexdigest()
            if doc_id in self.doc_ids:
                self.logger.debug(f"중복 문서: {item['title'][:30]}...")
                filtered_count += 1
                continue  # 예외 대신 스킵
            
            # 문서 ID 추가
            self.doc_ids.add(doc_id)
            item['doc_id'] = doc_id
            
            processed_items.append(item)
        
        # 필터링 비율 계산
        if items:
            filter_ratio = (filtered_count / len(items)) * 100
            self.logger.info(f"필터링 비율: {filter_ratio:.1f}% ({filtered_count}/{len(items)})")
        
        return processed_items

    def _check_keyword_conditions(self, text, keywords):
        """키워드 AND/OR 조건을 검사 (엄격한 AND 논리 적용)"""
        if not text or not keywords:
            return False

        # 지역 키워드 (항상 포함되어야 함)
        region_keyword = keywords[0]['text'] if keywords else None
        if not region_keyword or region_keyword.lower() not in text.lower():
            return False

        # 키워드가 1개만 있으면 (지역 키워드만) 무조건 통과
        if len(keywords) == 1:
            return True

        # AND 키워드가 있는지 확인
        and_keywords = [k['text'] for k in keywords[1:] if k['condition'] == 'AND' and k['text']]
        # OR 키워드가 있는지 확인
        or_keywords = [k['text'] for k in keywords[1:] if k['condition'] == 'OR' and k['text']]

        # AND 키워드가 없고 OR 키워드도 없으면, 지역 키워드만 있으므로 통과
        if not and_keywords and not or_keywords:
            return True

        # 모든 AND 키워드가 포함되어야 함
        for kw in and_keywords:
            if kw.lower() not in text.lower():
                return False

        # OR 키워드는 하나라도 포함되면 통과 (없으면 무시)
        if or_keywords:
            return any(kw.lower() in text.lower() for kw in or_keywords)
        else:
            return True

    def generate_keyword_variations(self, keywords):
        """키워드 조건에 따라 검색 쿼리 조합 생성"""
        if not keywords:
            return []
            
        # 지역 키워드 (항상 AND)
        region_keyword = keywords[0]['text'] if keywords else None
        if not region_keyword:
            return []
            
        # AND 키워드 (항상 포함)
        and_keywords = [k['text'] for k in keywords[1:] if k['condition'] == 'AND' and k['text']]
        
        # OR 키워드 (다양한 조합 생성)
        or_keywords = [k['text'] for k in keywords[1:] if k['condition'] == 'OR' and k['text']]
        
        variations = []
        
        # AND 키워드가 있는 경우, 무조건 포함
        if and_keywords:
            # 기본 검색어 (지역 + AND 키워드 모두 포함)
            base_query = f"{region_keyword} {' '.join(and_keywords)}"
            
            # OR 키워드가 없으면 기본 검색어만 사용
            if not or_keywords:
                variations.append(base_query)
            else:
                # OR 키워드가 있으면 각각 조합 추가
                for or_kw in or_keywords:
                    variations.append(f"{base_query} {or_kw}")
                
                # 모든 OR 키워드 한번에 추가 (선택적)
                if len(or_keywords) > 1:
                    variations.append(f"{base_query} {' '.join(or_keywords)}")
        else:
            # AND 키워드가 없는 경우
            # 지역 키워드만 사용
            variations.append(region_keyword)
            
            # OR 키워드가 있으면 각각 조합
            for or_kw in or_keywords:
                variations.append(f"{region_keyword} {or_kw}")
        
        # 무료 API 한도(100)를 고려하여 변형 수 제한
        max_variations = min(len(variations), max(1, self.max_daily_queries // 10))  # 페이지당 10개 결과
        if len(variations) > max_variations:
            self.logger.warning(f"API 쿼리 제한을 고려하여 키워드 변형을 {len(variations)}개에서 {max_variations}개로 제한합니다.")
            variations = variations[:max_variations]
        
        # 중복 제거 후 반환
        return list(set(variations))

    def crawl(self):
        """구글 검색 데이터 수집"""
        all_results = []
        
        try:
            # 크롤링 시작 시간 기록
            start_time = time.time()
            self.logger.info(f"====== 구글 검색 크롤링 시작: {time.strftime('%Y-%m-%d %H:%M:%S')} ======")
            self.logger.info(f"일일 API 쿼리 사용량: {self.query_count}/{self.max_daily_queries}")
            
            if isinstance(self.original_keywords[0], dict):
                keyword_info = [f"{k['text']}({'필수' if k.get('condition') == 'AND' else '선택적'})" for k in self.original_keywords]
                self.logger.info(f"검색 키워드: {keyword_info}")
            else:
                self.logger.info(f"검색 키워드: {self.keywords}")
            
            # 키워드 조합 생성
            keyword_variations = self.generate_keyword_variations(self.original_keywords)
            self.logger.info(f"생성된 검색 쿼리 ({len(keyword_variations)}개): {keyword_variations}")
            
            # 각 검색 쿼리에 대해 API 호출
            for i, query in enumerate(keyword_variations, 1):
                self.logger.info(f"\n>>> 검색 쿼리 {i}/{len(keyword_variations)}: '{query}' 처리 중...")
                query_results = []
                
                # 일일 쿼리 한도 확인
                if self.query_count >= self.max_daily_queries:
                    self.logger.warning(f"⚠️ 일일 API 쿼리 한도({self.max_daily_queries}개)에 도달했습니다. 나머지 쿼리는 처리하지 않습니다.")
                    break
                
                # 남은 쿼리 수를 고려하여 페이지 수 조정
                adjusted_max_pages = min(self.max_pages, max(1, self.max_daily_queries - self.query_count))
                if adjusted_max_pages < self.max_pages:
                    self.logger.warning(f"⚠️ API 쿼리 한도를 고려하여 페이지 수를 {self.max_pages}에서 {adjusted_max_pages}로 제한합니다.")
                
                # 여러 페이지 결과 수집
                for page in range(1, adjusted_max_pages + 1):
                    # API 요청 파라미터 설정
                    params = {
                        'key': self.api_key,
                        'cx': self.search_engine_id,
                        'q': query,
                        'start': (page - 1) * 10 + 1,  # 시작 인덱스
                        'num': 10                       # 결과 수 (최대 10)
                    }
                    
                    try:
                        # API 쿼리 카운트 증가 및 한도 확인
                        if not self._increment_query_count():
                            self.logger.warning("일일 API 쿼리 한도에 도달했습니다. 더 이상 요청하지 않습니다.")
                            break
                        
                        # API 요청
                        response = requests.get(self.base_url, params=params)
                        
                        # 응답 확인
                        if response.status_code != 200:
                            self.logger.error(f"API 오류 ({response.status_code}): {response.text}")
                            break
                            
                        # 결과 파싱
                        result_data = response.json()
                        items = result_data.get('items', [])
                        self.logger.info(f"페이지 {page}에서 {len(items)}개 결과 찾음")
                        
                        if not items:
                            self.logger.info(f"더 이상 결과가 없습니다. 페이지 {page-1}에서 종료")
                            break
                            
                        # 각 아이템 처리
                        for item in items:
                            # 기본 정보 추출
                            title = self._clean_html_content(item.get('title', ''))
                            snippet = self._clean_html_content(item.get('snippet', ''))
                            url = item.get('link', '')
                            
                            # 날짜 정보 파싱 시도
                            published_date = time.strftime("%Y%m%d")
                            try:
                                if 'pagemap' in item and 'metatags' in item['pagemap']:
                                    for metatag in item['pagemap']['metatags']:
                                        if 'article:published_time' in metatag:
                                            date_str = metatag['article:published_time'][:10]
                                            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                                            published_date = date_obj.strftime("%Y%m%d")
                                            break
                            except Exception as e:
                                self.logger.debug(f"날짜 파싱 오류: {e}")
                            
                            try:
                                date_obj = datetime.strptime(published_date, "%Y%m%d")
                            except:
                                date_obj = datetime.now()
                                
                            # 감성 분석
                            sentiment, confidence = self.analyze_text_sentiment(f"{title} {snippet}")
                            
                            # 결과 저장
                            result_item = {
                                "title": title,
                                "content": snippet,
                                "url": url,
                                "published_date": published_date,
                                "date_obj": date_obj.isoformat(),
                                "platform": "google",
                                "keyword": query,
                                "original_keywords": ",".join(self.keywords),
                                "sentiment": sentiment,
                                "confidence": confidence,
                                "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                            
                            query_results.append(result_item)
                        
                        # API 제한 방지를 위한 딜레이
                        time.sleep(random.uniform(1.0, 2.0))
                        
                    except Exception as e:
                        self.logger.error(f"쿼리 '{query}' 페이지 {page} 처리 중 오류: {str(e)}")
                        break
                
                # 후처리
                filtered_results = self._postprocess(query_results, self.original_keywords)
                self.logger.info(f"[요약] 쿼리 '{query}'에 대해 {len(query_results)}개 중 {len(filtered_results)}개 필터링됨")
                
                # 결과 저장
                if filtered_results:
                    # 결과 정렬 (날짜순)
                    filtered_results.sort(key=lambda x: x['date_obj'], reverse=True)
                    
                    # 파일 저장
                    keywords_str = '_'.join(self.keywords)
                    filename = f"google_{len(filtered_results)}_{keywords_str}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                    filepath = os.path.join(self.save_dir, filename)
                    os.makedirs(self.save_dir, exist_ok=True)
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(filtered_results, f, ensure_ascii=False, indent=2)
                        
                    self.logger.info(f"결과를 {filepath}에 저장했습니다.")
                    all_results.extend(filtered_results)
                else:
                    self.logger.warning(f"쿼리 '{query}'에 대한 저장할 결과가 없습니다.")
            
            # 최종 결과 통합 저장
            if all_results:
                # 중복 제거
                unique_results = {item['url']: item for item in all_results}.values()
                combined_results = list(unique_results)
                combined_results.sort(key=lambda x: x['date_obj'], reverse=True)
                
                # 통합 파일 저장
                keywords_str = '_'.join(self.keywords)
                combined_filename = f"google_combined_{len(combined_results)}_{keywords_str}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                combined_filepath = os.path.join(self.save_dir, combined_filename)
                
                with open(combined_filepath, "w", encoding="utf-8") as f:
                    json.dump(combined_results, f, ensure_ascii=False, indent=2)
                    
                self.logger.info(f"통합 결과 {len(combined_results)}개를 {combined_filepath}에 저장했습니다.")
                
                # 요약 정보 출력
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.logger.info(f"크롤링 종료: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(f"소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")
                self.logger.info(f"수집된 총 문서: {len(combined_results)}개")
                self.logger.info(f"사용된 Google API 쿼리: {self.query_count}/{self.max_daily_queries}")
                
                return combined_results
            else:
                self.logger.warning("수집된 결과가 없습니다.")
                return []
                
        except Exception as e:
            self.logger.error(f"크롤링 중 오류 발생: {str(e)}")
            return [] 