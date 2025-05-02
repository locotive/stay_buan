import argparse
import logging
from crawlers.naver_api_crawler import NaverSearchAPICrawler
# 나중에 다른 크롤러 임포트

def setup_logger():
    """로거 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("crawler.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("main")

def main():
    """메인 실행 함수"""
    logger = setup_logger()
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="부안군 감성 분석을 위한 데이터 수집기")
    parser.add_argument("--keywords", nargs="+", default=["부안", "변산반도", "부안여행"], 
                        help="검색할 키워드 목록 (공백으로 구분)")
    parser.add_argument("--platform", default="naver", 
                        choices=["naver"], 
                        help="데이터를 수집할 플랫폼")
    parser.add_argument("--pages", type=int, default=3, 
                        help="크롤링할 페이지 수")
    
    args = parser.parse_args()
    
    logger.info(f"Starting crawler with keywords: {args.keywords}, platform: {args.platform}, pages: {args.pages}")
    
    try:
        # 플랫폼에 따른 크롤러 선택
        if args.platform == "naver":
            crawler = NaverSearchAPICrawler(args.keywords, args.pages)
            results = crawler.crawl()
            logger.info(f"Collected {len(results)} items from Naver Search")
        else:
            logger.error(f"Unsupported platform: {args.platform}")
            return
            
    except Exception as e:
        logger.error(f"Error running crawler: {e}")
    
    logger.info("Crawling completed")

if __name__ == "__main__":
    main() 