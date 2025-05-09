import os
import requests
import time
import random
import json
from urllib.parse import quote
import logging
import hashlib
from datetime import datetime

from core.base_crawler import BaseCrawler
from utils.savers import DataSaver
from core.sentiment_analysis_ensemble import EnsembleSentimentAnalyzer

class YouTubeCrawler(BaseCrawler):
    """유튜브 크롤러 - 비디오 및 댓글 수집"""

    def __init__(self, keywords, max_results=50, max_comments=20, save_dir="data/raw", analyze_sentiment=True):
        """
        YouTube 크롤러 초기화
        
        Args:
            keywords: 검색할 키워드 리스트
            max_results: 수집할 최대 비디오 수 (기본값: 50)
            max_comments: 각 비디오당 수집할 최대 댓글 수
            save_dir: 저장 디렉터리
            analyze_sentiment: 감성 분석 수행 여부
        """
        super().__init__(keywords, save_dir=save_dir)
        self.max_results = max_results
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        
        if not self.api_key:
            self.logger.error("유튜브 API 키가 없습니다. 환경변수 YOUTUBE_API_KEY를 설정해주세요.")
            raise ValueError("유튜브 API 키가 없습니다.")
            
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.max_comments = max_comments
        self.doc_ids = set()  # 중복 문서 확인용
        self.analyze_sentiment = analyze_sentiment
        
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

    def get_video_details(self, video_ids):
        """비디오 상세 정보 가져오기"""
        if not video_ids:
            return []
            
        # 한 번에 최대 50개까지만 요청 가능
        chunks = [video_ids[i:i+50] for i in range(0, len(video_ids), 50)]
        all_videos = []
        
        for chunk in chunks:
            video_url = f"{self.base_url}/videos"
            params = {
                'part': 'snippet,statistics,contentDetails',
                'id': ','.join(chunk),
                'key': self.api_key
            }
            
            try:
                response = requests.get(video_url, params=params)
                if response.status_code == 200:
                    videos = response.json().get('items', [])
                    all_videos.extend(videos)
                else:
                    self.logger.error(f"비디오 상세 정보 요청 실패: {response.status_code} - {response.text}")
                
                # API 제한을 피하기 위한 대기
                time.sleep(random.uniform(0.5, 1.0))
            except Exception as e:
                self.logger.error(f"비디오 상세 정보 요청 중 오류: {str(e)}")
                
        return all_videos
        
    def get_video_comments(self, video_id):
        """비디오 댓글 가져오기"""
        comments = []
        comments_url = f"{self.base_url}/commentThreads"
        params = {
            'part': 'snippet',
            'videoId': video_id,
            'maxResults': self.max_comments,
            'key': self.api_key
        }
        
        try:
            response = requests.get(comments_url, params=params)
            if response.status_code == 200:
                comment_items = response.json().get('items', [])
                for item in comment_items:
                    comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    
                    # 댓글 감성 분석
                    sentiment, confidence = self.analyze_text_sentiment(comment_text)
                    
                    comments.append({
                        'text': comment_text,
                        'author': item['snippet']['topLevelComment']['snippet']['authorDisplayName'],
                        'published_at': item['snippet']['topLevelComment']['snippet']['publishedAt'],
                        'like_count': item['snippet']['topLevelComment']['snippet']['likeCount'],
                        'sentiment': sentiment,
                        'confidence': confidence
                    })
            elif response.status_code == 403 and "commentsDisabled" in response.text:
                self.logger.info(f"비디오 {video_id}의 댓글이 비활성화되어 있습니다.")
            else:
                self.logger.error(f"댓글 요청 실패: {response.status_code} - {response.text}")
            
            # API 제한을 피하기 위한 대기
            time.sleep(random.uniform(0.5, 1.0))
        except Exception as e:
            self.logger.error(f"댓글 요청 중 오류: {str(e)}")
            
        return comments

    def crawl(self):
        """유튜브 데이터 수집"""
        all_results = []
        
        try:
            # 크롤링 시작 시간 기록
            start_time = time.time()
            self.logger.info(f"====== 유튜브 크롤링 시작: {time.strftime('%Y-%m-%d %H:%M:%S')} ======")
            
            if isinstance(self.original_keywords[0], dict):
                keyword_info = [f"{k['text']}({'필수' if k.get('condition') == 'AND' else '선택적'})" for k in self.original_keywords]
                self.logger.info(f"검색 키워드: {keyword_info}")
            else:
                self.logger.info(f"검색 키워드: {self.keywords}")
            
            # 키워드별 검색
            for keyword in self.keywords:
                self.logger.info(f"키워드 '{keyword}' 처리 중...")
                encoded_keyword = quote(keyword)
                keyword_results = []
                
                # 1. 검색으로 비디오 ID 수집
                search_url = f"{self.base_url}/search"
                params = {
                    'part': 'snippet',
                    'q': encoded_keyword,
                    'type': 'video',
                    'maxResults': min(50, self.max_results),  # API 최대 50개까지
                    'order': 'relevance',  # 관련성 기준 정렬 (date, rating, viewCount 등 가능)
                    'key': self.api_key
                }
                
                try:
                    response = requests.get(search_url, params=params)
                    if response.status_code != 200:
                        self.logger.error(f"검색 요청 실패: {response.status_code} - {response.text}")
                        continue
                        
                    videos = response.json().get('items', [])
                    video_ids = [video['id']['videoId'] for video in videos]
                    
                    # 2. 비디오 상세 정보 가져오기
                    detailed_videos = self.get_video_details(video_ids)
                    
                    for video in detailed_videos:
                        video_id = video['id']
                        
                        # 3. 각 비디오의 댓글 가져오기
                        comments = self.get_video_comments(video_id)
                        
                        # 4. 데이터 구조화
                        try:
                            published_date = video['snippet']['publishedAt'][:10].replace('-', '')  # YYYYMMDD 형식
                        except:
                            published_date = time.strftime("%Y%m%d")
                            
                        try:
                            date_obj = datetime.strptime(published_date, "%Y%m%d")
                        except:
                            date_obj = datetime.now()
                            
                        # 고유 ID 생성
                        doc_id = hashlib.sha256(f"youtube_{video_id}".encode()).hexdigest()
                        
                        # 중복 확인
                        if doc_id in self.doc_ids:
                            continue
                        self.doc_ids.add(doc_id)
                        
                        # 비디오 제목과 설명의 감성 분석
                        title = video['snippet']['title']
                        description = video['snippet']['description']
                        combined_text = f"{title} {description}"
                        
                        sentiment, confidence = self.analyze_text_sentiment(combined_text)
                        
                        # 댓글 감성 분석 결과 집계
                        comment_sentiments = {
                            'positive': 0,
                            'neutral': 0,
                            'negative': 0
                        }
                        
                        for comment in comments:
                            if comment.get('sentiment'):
                                comment_sentiments[comment['sentiment']] += 1
                                
                        # 비디오와 댓글 감성 분석 결과 종합
                        overall_sentiment = sentiment
                        # 댓글이 많으면 댓글 감성도 고려
                        if sum(comment_sentiments.values()) > 5 and sentiment:
                            # 비디오와 댓글 감성 가중치 조정 (비디오 60%, 댓글 40%)
                            comment_majority = max(comment_sentiments, key=comment_sentiments.get)
                            if comment_majority != sentiment and comment_sentiments[comment_majority] > sum(comment_sentiments.values()) * 0.6:
                                overall_sentiment = comment_majority
                        
                        video_data = {
                            'doc_id': doc_id,
                            'title': title,
                            'content': description,
                            'url': f"https://www.youtube.com/watch?v={video_id}",
                            'platform': 'youtube',
                            'keyword': keyword,
                            'original_keywords': ','.join(self.keywords),
                            'published_date': published_date,
                            'date_obj': date_obj.isoformat(),
                            'channel_title': video['snippet']['channelTitle'],
                            'view_count': video['statistics'].get('viewCount', '0'),
                            'like_count': video['statistics'].get('likeCount', '0'),
                            'comment_count': video['statistics'].get('commentCount', '0'),
                            'duration': video['contentDetails']['duration'],
                            'comments': comments,
                            'sentiment': overall_sentiment,
                            'confidence': confidence,
                            'comment_sentiments': comment_sentiments,
                            'crawled_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        keyword_results.append(video_data)
                        
                except Exception as e:
                    self.logger.error(f"키워드 '{keyword}' 처리 중 오류: {str(e)}")
                
                # 키워드별 결과 저장
                if keyword_results:
                    filename = f"youtube_{len(keyword_results)}_{keyword}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                    filepath = os.path.join(self.save_dir, filename)
                    os.makedirs(self.save_dir, exist_ok=True)
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(keyword_results, f, ensure_ascii=False, indent=2)
                        
                    self.logger.info(f"키워드 '{keyword}'에 대해 {len(keyword_results)}개 비디오 저장 완료: {filepath}")
                    all_results.extend(keyword_results)
                else:
                    self.logger.warning(f"키워드 '{keyword}'에 대한 결과가 없습니다.")
            
            # 크롤링 종료 시간 기록 및 요약
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.logger.info(f"크롤링 종료: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")
            self.logger.info(f"수집된 총 비디오: {len(all_results)}개")
            
            # 모든 결과 통합 저장
            if all_results:
                combined_filename = f"youtube_combined_{len(all_results)}_{'_'.join(self.keywords)}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                combined_filepath = os.path.join(self.save_dir, combined_filename)
                
                with open(combined_filepath, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                    
                self.logger.info(f"통합 결과 저장 완료: {combined_filepath}")
                
        except Exception as e:
            self.logger.error(f"크롤링 중 오류 발생: {str(e)}")
            
        return all_results

    def generate_filename(self, keyword):
        """파일명 생성"""
        return f"youtube_{keyword}_{time.strftime('%Y%m%d_%H%M%S')}.json"
