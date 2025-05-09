import argparse
import logging
import os
import time
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv
from crawlers.naver_api_crawler import NaverSearchAPICrawler
from crawlers.youtube import YouTubeCrawler
from crawlers.google_search import GoogleSearchCrawler
from crawlers.fmkorea import FMKoreaCrawler
from crawlers.dcinside import DCInsideCrawler
from crawlers.buan_gov import BuanGovCrawler
# 나중에 다른 크롤러 임포트

# .env 파일 로드
load_dotenv()

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

def parse_keywords(keywords_list):
    """키워드 목록을 NaverSearchAPICrawler가 요구하는 형식으로 변환"""
    if not keywords_list:
        return []
        
    # 첫 번째 키워드는 지역 키워드로 처리
    parsed_keywords = [{"text": keywords_list[0], "condition": "AND"}]
    
    # 나머지 키워드는 OR 조건으로 처리
    for kw in keywords_list[1:]:
        parsed_keywords.append({"text": kw, "condition": "OR"})
        
    return parsed_keywords

def crawl_platform(platform, keywords, max_pages, max_comments, no_sentiment, browser_type, max_daily_queries):
    """특정 플랫폼 크롤링 실행"""
    try:
        logger.info(f"\n{'='*20} {platform.upper()} 크롤링 시작 {'='*20}")
        logger.info(f"키워드: {keywords}")
        logger.info(f"최대 페이지: {max_pages}")
        logger.info(f"최대 댓글: {max_comments}")
        logger.info(f"감성 분석: {'비활성화' if no_sentiment else '활성화'}")
        logger.info(f"브라우저: {browser_type}")

        # 네이버 API 키 체크
        if platform == "naver":
            if not os.getenv("NAVER_CLIENT_ID") or not os.getenv("NAVER_CLIENT_SECRET"):
                logger.error("""
                NAVER_CLIENT_ID 또는 NAVER_CLIENT_SECRET 환경변수가 설정되지 않았습니다.
                .env 파일에 다음 내용을 추가해주세요:
                NAVER_CLIENT_ID=your_client_id
                NAVER_CLIENT_SECRET=your_client_secret
                """)
                return platform, []

        # 유튜브 API 키 체크
        elif platform == "youtube":
            if not os.getenv("YOUTUBE_API_KEY"):
                logger.error("""
                YOUTUBE_API_KEY 환경변수가 설정되지 않았습니다.
                .env 파일에 다음 내용을 추가해주세요:
                YOUTUBE_API_KEY=your_api_key
                """)
                return platform, []

        # 구글 API 키 체크
        elif platform == "google":
            if not os.getenv("GOOGLE_API_KEY"):
                logger.error("""
                GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.
                .env 파일에 다음 내용을 추가해주세요:
                GOOGLE_API_KEY=your_api_key
                """)
                return platform, []

        # 크롤러 초기화 및 실행
        start_time = time.time()
        results = []

        if platform == "naver":
            crawler = NaverSearchAPICrawler(keywords, max_pages=max_pages, save_dir="data/raw", analyze_sentiment=not no_sentiment, browser_type=browser_type)
            results = crawler.crawl()
            logger.info(f"네이버에서 {len(results)}개 항목 수집 완료")

        elif platform == "youtube":
            crawler = YouTubeCrawler(
                keywords,
                max_results=max_pages * 10,  # 페이지당 약 10개 결과
                max_comments=max_comments,
                save_dir="data/raw",
                analyze_sentiment=not no_sentiment
            )
            results = crawler.crawl()
            logger.info(f"유튜브에서 {len(results)}개 비디오 수집 완료")

        elif platform == "google":
            crawler = GoogleSearchCrawler(keywords, max_pages=max_pages, save_dir="data/raw", analyze_sentiment=not no_sentiment, max_daily_queries=max_daily_queries)
            results = crawler.crawl()
            logger.info(f"구글에서 {len(results)}개 항목 수집 완료")

        elif platform == "dcinside":
            crawler = DCInsideCrawler(keywords, max_pages=max_pages, max_comments=max_comments, save_dir="data/raw", analyze_sentiment=not no_sentiment, browser_type=browser_type, respect_robots=False)
            results = crawler.crawl()
            logger.info(f"디시인사이드에서 {len(results)}개 게시글 수집 완료")

        elif platform == "fmkorea":
            crawler = FMKoreaCrawler(keywords, max_pages=max_pages, max_comments=max_comments, save_dir="data/raw", analyze_sentiment=not no_sentiment, browser_type=browser_type, respect_robots=False)
            results = crawler.crawl()
            logger.info(f"FM코리아에서 {len(results)}개 게시글 수집 완료")

        elif platform == "buan":
            crawler = BuanGovCrawler(keywords, max_pages=max_pages, save_dir="data/raw", analyze_sentiment=not no_sentiment, browser_type="firefox")
            results = crawler.crawl()
            logger.info(f"부안군에서 {len(results)}개 게시글 수집 완료")

        else:
            logger.error(f"지원하지 않는 플랫폼: {platform}")
            return platform, []

        end_time = time.time()
        logger.info(f"\n{'='*20} {platform.upper()} 크롤링 완료 {'='*20}")
        logger.info(f"소요 시간: {end_time - start_time:.2f}초")
        logger.info(f"수집된 항목 수: {len(results)}개")

        return platform, results

    except Exception as e:
        logger.error(f"{platform} 크롤링 중 오류 발생: {str(e)}")
        logger.exception("상세 오류 정보:")
        return platform, []

def save_combined_results(all_platform_results, save_dir="data/raw", keywords=None):
    """모든 플랫폼의 결과를 통합하여 저장"""
    logger.info("\n" + "="*50)
    logger.info("결과 통합 및 저장 시작")
    
    # 평탄화된 결과 리스트
    flattened_results = []
    platform_counts = {}
    
    for platform, results in all_platform_results.items():
        if results:  # 결과가 있는 경우만 처리
            flattened_results.extend(results)
            platform_counts[platform] = len(results)
            logger.info(f"{platform.upper()}: {len(results)}개 항목")
    
    if not flattened_results:
        logger.warning("저장할 결과가 없습니다.")
        return None
        
    # 결과 저장
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    keywords_str = "_".join(keywords) if keywords else "all"
    
    # 통합 파일 경로
    os.makedirs(save_dir, exist_ok=True)
    combined_filename = f"combined_{len(flattened_results)}_{keywords_str}_{timestamp}.json"
    combined_filepath = os.path.join(save_dir, combined_filename)
    
    # JSON으로 저장
    import json
    with open(combined_filepath, "w", encoding="utf-8") as f:
        json.dump(flattened_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"통합 결과 저장 완료: {combined_filepath}")
    logger.info("="*50)
        
    return {
        "path": combined_filepath,
        "total": len(flattened_results),
        "platform_counts": platform_counts
    }

def print_crawler_summary(summary, start_time, logger):
    """크롤링 결과 요약 출력"""
    if not summary:
        logger.warning("수집된 결과가 없습니다.")
        return
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info("\n" + "="*50)
    logger.info("크롤링 최종 결과 요약")
    logger.info(f"시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    logger.info(f"종료 시간: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    logger.info(f"총 소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")
    logger.info(f"총 수집 항목: {summary['total']}개")
    
    # 플랫폼별 결과 수
    logger.info("\n플랫폼별 수집 결과:")
    for platform, count in summary['platform_counts'].items():
        logger.info(f"- {platform.upper()}: {count}개")
    
    logger.info(f"\n통합 결과 파일: {summary['path']}")
    logger.info("="*50)

def main():
    """메인 함수"""
    # 로거 초기화
    global logger
    logger = setup_logger()
    
    # 시작 시간 기록
    start_time = time.time()

    parser = argparse.ArgumentParser(description="부안 관련 데이터 크롤러")
    parser.add_argument("--platform", type=str, default="all", help="크롤링할 플랫폼 (all, naver, youtube, google, dcinside, fmkorea, buan)")
    parser.add_argument("--keywords", type=str, nargs="+", required=True, help="검색할 키워드")
    parser.add_argument("--max-pages", type=int, default=5, help="수집할 최대 페이지 수")
    parser.add_argument("--max-comments", type=int, default=30, help="각 게시글당 수집할 최대 댓글 수")
    parser.add_argument("--no-sentiment", action="store_true", help="감성 분석 비활성화")
    parser.add_argument("--browser", type=str, default="chrome", choices=["chrome", "firefox"], help="사용할 브라우저 (chrome 또는 firefox)")
    parser.add_argument("--loose-filter", action="store_true", help="키워드 필터링 완화 (OR 조건 사용)")
    parser.add_argument("--respect-robots", action="store_true", help="robots.txt 정책 준수")
    parser.add_argument("--parallel", action="store_true", help="병렬 처리 활성화")
    parser.add_argument("--max-workers", type=int, default=4, help="최대 워커 수 (1-8)")
    parser.add_argument("--max-daily-queries", type=int, default=100, help="구글 API 일일 최대 쿼리 수 (기본값: 100)")
    args = parser.parse_args()

    logger.info("\n" + "="*50)
    logger.info("크롤링 시작")
    logger.info(f"시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"검색 키워드: {args.keywords}")
    logger.info(f"브라우저: {args.browser}")
    logger.info(f"필터링 모드: {'완화 (OR 조건)' if args.loose_filter else '엄격 (AND 조건)'}")
    logger.info(f"robots.txt 정책: {'준수' if args.respect_robots else '무시'}")
    logger.info(f"병렬 처리: {'활성화' if args.parallel else '비활성화'}")
    if args.parallel:
        logger.info(f"최대 워커 수: {args.max_workers}")
    logger.info("="*50)

    # 키워드 형식 변환
    keywords = [{"text": k, "condition": "OR" if args.loose_filter else "AND"} for k in args.keywords]

    # 플랫폼 목록 설정
    platforms = ["naver", "youtube", "google", "dcinside", "fmkorea", "buan"] if args.platform == "all" else args.platform.split(",")
    logger.info(f"크롤링 대상 플랫폼: {', '.join(platforms)}")

    all_platform_results = {}

    # 병렬 처리 설정
    if args.parallel:
        max_workers = min(args.max_workers, 8)  # 최대 8개로 제한
        logger.info(f"병렬 처리 시작 (워커 수: {max_workers})")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for platform in platforms:
                future = executor.submit(
                    crawl_platform,
                    platform,
                    keywords,
                    args.max_pages,
                    args.max_comments,
                    args.no_sentiment,
                    args.browser,
                    args.max_daily_queries
                )
                futures.append(future)

            # 결과 수집
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="크롤링 진행률"):
                platform, results = future.result()
                if results:
                    all_platform_results[platform] = results

    else:
        # 순차 처리
        logger.info("순차 처리 시작")
        for platform in platforms:
            platform, results = crawl_platform(platform, keywords, args.max_pages, args.max_comments, args.no_sentiment, args.browser, args.max_daily_queries)
            if results:
                all_platform_results[platform] = results

    # 결과 저장 및 요약 출력
    summary = save_combined_results(all_platform_results, "data/raw", args.keywords)
    print_crawler_summary(summary, start_time, logger)

    logger.info("\n" + "="*50)
    logger.info("크롤링 종료")
    logger.info(f"종료 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*50)

if __name__ == "__main__":
    main() 