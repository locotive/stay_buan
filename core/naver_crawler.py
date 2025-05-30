from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .base_crawler import BaseCrawler
from utils.data_processor import DataProcessor
from utils.data_normalizer import PlatformDataNormalizer
from dashboard.crawler.crawler_status import CrawlerStatus

logger = logging.getLogger(__name__)

class NaverCrawler(BaseCrawler):
    """네이버 크롤러 클래스"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.data_processor = DataProcessor()
        self.data_normalizer = PlatformDataNormalizer()
        self.status = CrawlerStatus()
        
    async def crawl(self, keyword: str, start_date: str, end_date: str) -> List[Dict]:
        """네이버 크롤링 실행"""
        try:
            # 크롤링 시작 상태 설정
            platforms = ["네이버 뉴스", "네이버 블로그", "네이버 카페"]
            self.status.start_crawling(platforms)
            
            all_data = []
            
            # 네이버 뉴스 크롤링
            self.status.update_platform("네이버 뉴스")
            news_data = await self._crawl_news(keyword, start_date, end_date)
            if news_data:
                all_data.extend(news_data)
            self.status.complete_platform("네이버 뉴스")
            
            # 네이버 블로그 크롤링
            self.status.update_platform("네이버 블로그")
            blog_data = await self._crawl_blog(keyword, start_date, end_date)
            if blog_data:
                all_data.extend(blog_data)
            self.status.complete_platform("네이버 블로그")
            
            # 네이버 카페 크롤링
            self.status.update_platform("네이버 카페")
            cafe_data = await self._crawl_cafe(keyword, start_date, end_date)
            if cafe_data:
                all_data.extend(cafe_data)
            self.status.complete_platform("네이버 카페")
            
            # 크롤링 완료
            self.status.complete_crawling()
            return all_data
            
        except Exception as e:
            error_msg = f"네이버 크롤링 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            self.status.set_error(error_msg)
            raise
            
    async def _crawl_news(self, keyword: str, start_date: str, end_date: str) -> List[Dict]:
        """네이버 뉴스 크롤링"""
        try:
            total_pages = 100  # 예시 페이지 수
            news_data = []
            
            for page in range(1, total_pages + 1):
                # 진행률 업데이트
                progress = (page / total_pages) * 100
                self.status.update_subtask("네이버 뉴스", "뉴스 검색", progress)
                
                # 뉴스 검색 API 호출
                search_results = await self._search_news(keyword, page)
                if not search_results:
                    break
                    
                # 상세 수집을 병렬로 수행
                tasks = []
                for idx, result in enumerate(search_results):
                    detail_progress = (idx / len(search_results)) * 100
                    self.status.update_subtask("네이버 뉴스", "뉴스 상세 수집", detail_progress)
                    tasks.append(self._get_news_detail(result['link']))

                details = await asyncio.gather(*tasks, return_exceptions=True)
                for detail in details:
                    if isinstance(detail, Exception):
                        logger.warning(f"뉴스 상세 수집 중 예외 발생: {detail}")
                    elif detail:
                        news_data.append(detail)
                        
                # API 호출 간격 조절
                await self._wait()
                
            return news_data
            
        except Exception as e:
            logger.error(f"네이버 뉴스 크롤링 중 오류 발생: {str(e)}")
            raise
            
    async def _crawl_blog(self, keyword: str, start_date: str, end_date: str) -> List[Dict]:
        """네이버 블로그 크롤링"""
        try:
            total_pages = 100  # 예시 페이지 수
            blog_data = []
            
            for page in range(1, total_pages + 1):
                # 진행률 업데이트
                progress = (page / total_pages) * 100
                self.status.update_subtask("네이버 블로그", "블로그 검색", progress)
                
                # 블로그 검색 API 호출
                search_results = await self._search_blog(keyword, page)
                if not search_results:
                    break
                    
                # 상세 수집을 병렬로 수행
                tasks = []
                for idx, result in enumerate(search_results):
                    detail_progress = (idx / len(search_results)) * 100
                    self.status.update_subtask("네이버 블로그", "블로그 상세 수집", detail_progress)
                    tasks.append(self._get_blog_detail(result['link']))

                details = await asyncio.gather(*tasks, return_exceptions=True)
                for detail in details:
                    if isinstance(detail, Exception):
                        logger.warning(f"블로그 상세 수집 중 예외 발생: {detail}")
                    elif detail:
                        blog_data.append(detail)
                        
                # API 호출 간격 조절
                await self._wait()
                
            return blog_data
            
        except Exception as e:
            logger.error(f"네이버 블로그 크롤링 중 오류 발생: {str(e)}")
            raise
            
    async def _crawl_cafe(self, keyword: str, start_date: str, end_date: str) -> List[Dict]:
        """네이버 카페 크롤링"""
        try:
            total_pages = 100  # 예시 페이지 수
            cafe_data = []
            
            for page in range(1, total_pages + 1):
                # 진행률 업데이트
                progress = (page / total_pages) * 100
                self.status.update_subtask("네이버 카페", "카페 검색", progress)
                
                # 카페 검색 API 호출
                search_results = await self._search_cafe(keyword, page)
                if not search_results:
                    break
                    
                # 상세 수집을 병렬로 수행
                tasks = []
                for idx, result in enumerate(search_results):
                    detail_progress = (idx / len(search_results)) * 100
                    self.status.update_subtask("네이버 카페", "카페 상세 수집", detail_progress)
                    tasks.append(self._get_cafe_detail(result['link']))

                details = await asyncio.gather(*tasks, return_exceptions=True)
                for detail in details:
                    if isinstance(detail, Exception):
                        logger.warning(f"카페 상세 수집 중 예외 발생: {detail}")
                    elif detail:
                        cafe_data.append(detail)
                        
                # API 호출 간격 조절
                await self._wait()
                
            return cafe_data
            
        except Exception as e:
            logger.error(f"네이버 카페 크롤링 중 오류 발생: {str(e)}")
            raise
            
    async def _wait(self):
        """API 호출 간격 조절"""
        await asyncio.sleep(self.config.get('delay', 1.0))
        
    # ... (기존 메서드들은 그대로 유지) 