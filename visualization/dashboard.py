import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from utils.data_loader import DataLoader
from utils.data_normalizer import load_and_normalize_data
from utils.data_processor import DataProcessor
from core.sentiment_analysis import SentimentAnalyzer
from core.sentiment_analysis_kobert import KoBERTSentimentAnalyzer
from core.sentiment_analysis_ensemble import EnsembleSentimentAnalyzer
from reporting.report_generator_gpt import GPTReportGenerator
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from folium.plugins import MarkerCluster
from reporting.pdf_report_generator import PDFReportGenerator
from crawlers.naver_api_crawler import NaverSearchAPICrawler
from crawlers.youtube import YouTubeCrawler
from crawlers.google_search import GoogleSearchCrawler
import os
import glob
from collections import Counter
from dateutil import parser
import json
import random
import subprocess
import threading
import sys
from pathlib import Path
import re
import time
import queue
import logging
import concurrent.futures

# 로깅 설정
os.makedirs("data/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/logs/streamlit_dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dashboard")

# 크롤링 상태를 파일로 저장하고 읽어오는 함수
def save_crawler_status(status):
    """크롤링 상태를 파일로 저장"""
    try:
        os.makedirs("data/status", exist_ok=True)
        with open("data/status/crawler_status.json", "w") as f:
            json.dump(status, f)
        logger.info("크롤링 상태 저장됨")
    except Exception as e:
        logger.error(f"크롤링 상태 저장 중 오류: {str(e)}", exc_info=True)

def load_crawler_status():
    """파일에서 크롤링 상태 읽기"""
    try:
        if os.path.exists("data/status/crawler_status.json"):
            with open("data/status/crawler_status.json", "r") as f:
                status = json.load(f)
            logger.info("크롤링 상태 로드됨")
            return status
        else:
            return {
                'message': '',
                'progress': 0.0,
                'result': '',
                'is_running': False,
                'update_timestamp': 0,
                'command': ''
            }
    except Exception as e:
        logger.error(f"크롤링 상태 로드 중 오류: {str(e)}", exc_info=True)
        return {
            'message': '',
            'progress': 0.0,
            'result': '',
            'is_running': False,
            'update_timestamp': 0,
            'command': ''
        }

# 크롤링 상태를 저장할 전역 변수
crawler_status = load_crawler_status()

# 크롤링 스레드 함수
def run_crawler(cmd):
    """크롤링 명령 실행 (스레드 안전)"""
    global crawler_status
    
    try:
        # 초기 상태 업데이트
        crawler_status['message'] = "크롤링 실행 중..."
        crawler_status['progress'] = 0.0
        crawler_status['result'] = ""
        crawler_status['is_running'] = True
        crawler_status['update_timestamp'] = time.time()
        crawler_status['command'] = cmd
        save_crawler_status(crawler_status)  # 상태 저장
        
        logger.info(f"크롤링 명령 실행: {cmd}")
        
        with subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            shell=True
        ) as process:
            # 플랫폼별 진행 상황 표시 - 명령어에서 플랫폼 정보 추출
            all_platforms = ["naver", "youtube", "google", "dcinside", "fmkorea", "buan"]
            
            # 명령어에서 platform 파라미터 찾기
            platform_param = ""
            for part in cmd.split():
                if part.startswith("--platform"):
                    platform_param = part.split("=")[1] if "=" in part else cmd.split()[cmd.split().index(part) + 1]
                    break
            
            # 플랫폼 목록 결정
            if platform_param == "all" or not platform_param:
                platforms = all_platforms
            else:
                platforms = platform_param.split(",")
                
            progress_per_platform = 1.0 / len(platforms)
            current_platform = None
            platform_index = 0
            
            lines = []
            
            logger.info("크롤링 프로세스 시작됨, 출력 모니터링 중...")
            
            for line in process.stdout:
                line = line.strip()
                lines.append(line)
                logger.info(f"크롤링 출력: {line}")
                
                # 진행률 업데이트
                if "크롤링 시작" in line:
                    for platform in platforms:
                        if platform.upper() in line:
                            current_platform = platform
                            platform_index += 1
                            status_msg = f"현재 플랫폼: {current_platform.upper()} 크롤링 중... ({platform_index}/{len(platforms)})"
                            crawler_status['message'] = status_msg
                            crawler_status['progress'] = min(0.99, platform_index * progress_per_platform)
                            crawler_status['update_timestamp'] = time.time()
                            save_crawler_status(crawler_status)  # 상태 저장
                            logger.info(status_msg)
                            break
                
                # 최종 결과 확인
                if "통합 결과 저장 경로" in line:
                    result_path = line.split("통합 결과 저장 경로:")[1].strip()
                    crawler_status['result'] = f"✅ 크롤링 완료! 결과 저장 경로: {result_path}"
                    crawler_status['update_timestamp'] = time.time()
                    save_crawler_status(crawler_status)  # 상태 저장
                    logger.info(f"크롤링 완료, 결과 저장 경로: {result_path}")
            
            # 프로세스가 완료될 때까지 기다림
            return_code = process.wait()
            logger.info(f"크롤링 프로세스 종료, 리턴 코드: {return_code}")
            
            # 크롤링 완료
            crawler_status['progress'] = 1.0
            crawler_status['message'] = "✅ 크롤링 완료!"
            crawler_status['is_running'] = False
            crawler_status['update_timestamp'] = time.time()
            save_crawler_status(crawler_status)  # 상태 저장
            logger.info("크롤링 상태 업데이트: 완료")
            
            # 전체 로그 파일로 저장
            log_filename = f"crawl_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            os.makedirs("data/logs", exist_ok=True)
            with open(f"data/logs/{log_filename}", "w") as f:
                f.write("\n".join(lines))
            logger.info(f"크롤링 로그 저장: {log_filename}")
    
    except Exception as e:
        error_msg = f"❌ 크롤링 중 오류 발생: {str(e)}"
        crawler_status['message'] = error_msg
        crawler_status['progress'] = 1.0
        crawler_status['is_running'] = False
        crawler_status['update_timestamp'] = time.time()
        save_crawler_status(crawler_status)  # 상태 저장
        logger.error(error_msg, exc_info=True)
        
        # 오류 로그 저장
        error_log_filename = f"crawl_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs("data/logs", exist_ok=True)
        with open(f"data/logs/{error_log_filename}", "w") as f:
            f.write(f"오류 발생: {str(e)}\n")
            f.write(f"명령어: {cmd}\n")
            import traceback
            f.write(traceback.format_exc())
        logger.info(f"오류 로그 저장: {error_log_filename}")

def get_keywords_string(region_keyword, additional_keywords):
    """키워드 목록을 명령줄 인자 형식으로 변환"""
    keywords = [region_keyword]
    for kw in additional_keywords:
        if kw["text"].strip():
            keywords.append(kw["text"].strip())
    return ' '.join([f'"{k}"' if ' ' in k else k for k in keywords])

# 크롤링 결과 파일을 기반으로 샘플 데이터 로드
def load_latest_results(num_files=5):
    """최근 크롤링 결과 파일들 로드"""
    result_files = []
    
    # 플랫폼별 최근 파일 검색
    platforms = ["naver", "youtube", "google", "dcinside", "fmkorea", "buan", "combined"]
    
    for platform in platforms:
        files = glob.glob(f"data/raw/{platform}*.json")
        # 파일 수정 시간으로 정렬
        files.sort(key=os.path.getmtime, reverse=True)
        # 최근 파일 선택
        for file in files[:num_files]:
            if os.path.exists(file):
                try:
                    stats = os.stat(file)
                    mtime = datetime.fromtimestamp(stats.st_mtime)
                    
                    # 파일명에서 정보 추출
                    filename = os.path.basename(file)
                    match = re.search(r'(\w+)_(\d+)_([^_]+)', filename)
                    
                    platform_name = ""
                    item_count = 0
                    keywords = ""
                    
                    if match:
                        platform_name = match.group(1)
                        item_count = int(match.group(2))
                        keywords = match.group(3)
                    else:
                        # 정규식으로 추출할 수 없으면 파일명으로 대체
                        platform_name = platform
                        item_count = 0
                        keywords = filename
                    
                    result_files.append({
                        "filename": filename,
                        "platform": platform_name,
                        "keywords": keywords,
                        "items": item_count,
                        "path": file,
                        "modified": mtime.strftime("%Y-%m-%d %H:%M:%S")
                    })
                except Exception as e:
                    st.error(f"파일 정보 읽기 실패: {str(e)}")
    
    # 시간순 정렬
    result_files.sort(key=lambda x: x["modified"], reverse=True)
    return result_files[:num_files]

def get_latest_result_stats():
    """최근 크롤링 결과 통계"""
    try:
        result_files = glob.glob("data/raw/combined_*.json")
        if not result_files:
            return None
            
        # 가장 최근 파일
        latest_file = max(result_files, key=os.path.getmtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 플랫폼별 항목 수
        platform_counts = {}
        for item in data:
            platform = item.get('platform', 'unknown')
            platform_counts[platform] = platform_counts.get(platform, 0) + 1
        
        # 감성별 항목 수
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0, 'unknown': 0}
        for item in data:
            sentiment = item.get('sentiment', 'unknown')
            if sentiment == 0 or sentiment == 'negative':
                sentiment_counts['negative'] += 1
            elif sentiment == 1 or sentiment == 'neutral':
                sentiment_counts['neutral'] += 1
            elif sentiment == 2 or sentiment == 'positive':
                sentiment_counts['positive'] += 1
            else:
                sentiment_counts['unknown'] += 1
        
        # 날짜별 항목 수
        date_counts = {}
        for item in data:
            date = item.get('published_date', '')
            if date and len(date) >= 8:  # 'YYYYMMDD' 형식
                date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
            else:
                date = 'unknown'
            date_counts[date] = date_counts.get(date, 0) + 1
        
        return {
            'total': len(data),
            'filename': os.path.basename(latest_file),
            'platform_counts': platform_counts,
            'sentiment_counts': sentiment_counts,
            'date_counts': date_counts,
            'path': latest_file
        }
    except Exception as e:
        st.error(f"결과 통계 계산 중 오류 발생: {str(e)}")
        return None

@st.cache_data
def analyze_sentiment(text, analyzer):
    """텍스트 감성 분석 결과 캐싱"""
    sentiment, confidence = analyzer.predict(text)
    sentiment_label = 'negative' if sentiment == 0 else 'neutral' if sentiment == 1 else 'positive'
    return sentiment_label, confidence

def create_map(data):
    """지도 시각화 생성"""

    # None 이거나 비어있는 경우 종료
    if data is None:
        return None

    # 리스트인 경우 DataFrame으로 변환
    if isinstance(data, list):
        data = pd.DataFrame(data)

    # DataFrame이 아닌 경우 종료
    if not isinstance(data, pd.DataFrame):
        return None

    # DataFrame이 비어있는 경우 종료
    if data.empty:
        return None

    # 위치 정보가 있는 경우
    if 'location' in data.columns:
        locations = data['location'].dropna().unique()
        if len(locations) > 0:
            try:
                lat, lon = map(float, locations[0].split(','))
                return folium.Map(
                    location=[lat, lon],
                    zoom_start=13,
                    tiles='CartoDB positron'
                )
            except ValueError:
                pass  # location 형식이 이상할 경우 무시하고 기본 지도 생성

    # 위치 정보가 없는 경우: 부안군 중심
    return folium.Map(
        location=[35.7284, 126.7320],
        zoom_start=11,
        tiles='CartoDB positron'
    )

def get_available_datasets():
    """사용 가능한 데이터셋 목록을 반환"""
    datasets = []
    
    # 네이버 데이터셋
    for file in glob.glob("data/raw/naver_*.json"):
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) >= 4:
            count = parts[2]
            keywords = '_'.join(parts[3:-2])
            datasets.append({
                'filename': filename,
                'count': count,
                'keywords': keywords,
                'path': file,
                'platform': 'naver'
            })
    
    # 유튜브 데이터셋
    for file in glob.glob("data/raw/youtube_*.json"):
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) >= 4:
            count = parts[1]  # youtube_[count]_[keywords]_[timestamp].json
            keywords = '_'.join(parts[2:-2])
            datasets.append({
                'filename': filename,
                'count': count,
                'keywords': keywords,
                'path': file,
                'platform': 'youtube'
            })
    
    # 기타 플랫폼 (확장성 고려)
    for platform in ['google', 'dcinside', 'fmkorea', 'buan']:
        for file in glob.glob(f"data/raw/{platform}_*.json"):
            filename = os.path.basename(file)
            parts = filename.split('_')
            if len(parts) >= 3:
                count = parts[1]
                keywords = '_'.join(parts[2:-2])
                datasets.append({
                    'filename': filename,
                    'count': count,
                    'keywords': keywords,
                    'path': file,
                    'platform': platform
                })
    
    return sorted(datasets, key=lambda x: x['count'], reverse=True)

def load_selected_dataset(filepath):
    """선택된 데이터셋 로드 및 정규화"""
    try:
        # 데이터 로더 초기화
        data_loader = DataLoader()
        data_processor = DataProcessor()
        
        # 파일명에서 플랫폼 추출
        platform = os.path.basename(filepath).split('_')[0]
        
        # 데이터 로드 및 정규화
        df = load_and_normalize_data(filepath)
        
        # 데이터 처리 파이프라인 실행
        df = data_processor.process_data(df, platform)
        
        # 처리된 데이터 저장
        processed_filepath = data_processor.save_processed_data(df, platform)
        
        st.success(f"데이터 처리 완료: {processed_filepath}")
        return df
        
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return None

def main():
    """대시보드 메인 함수"""
    global crawler_status
    
    # 시작 시 크롤링 상태 파일에서 로드
    crawler_status = load_crawler_status()
    
    st.title("부안군 감성 분석 대시보드")
    
    # 세션 상태 초기화
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    if 'analyzer_option' not in st.session_state:
        st.session_state.analyzer_option = "Naive Bayes"
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'region_keyword' not in st.session_state:
        st.session_state.region_keyword = "부안"
    if 'additional_keywords' not in st.session_state:
        st.session_state.additional_keywords = [{"text": "", "condition": "AND"}]
    if 'dashboard_mode' not in st.session_state:
        st.session_state.dashboard_mode = "크롤링"
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = 0
    
    # 사이드바에서 대시보드 모드 선택
    with st.sidebar:
        st.header("대시보드 모드")
        dashboard_mode = st.radio(
            "모드 선택",
            ["크롤링", "데이터 분석"],
            index=0 if st.session_state.dashboard_mode == "크롤링" else 1,
            help="크롤링 모드: 다양한 플랫폼에서 데이터를 수집합니다. 데이터 분석 모드: 수집된 데이터를 분석합니다."
        )
        st.session_state.dashboard_mode = dashboard_mode
        
        st.markdown("---")
        
        # 모드에 따라 사이드바 내용 변경
        if dashboard_mode == "크롤링":
            st.header("크롤링 설정")
            
            # 지역 키워드 (필수)
            region_keyword = st.text_input("지역 키워드 (필수)", value=st.session_state.region_keyword, key="crawl_region_keyword")
            if region_keyword.strip():
                st.session_state.region_keyword = region_keyword
            
            # 추가 키워드 섹션
            st.markdown("##### 추가 키워드")
            
            # 키워드 입력 필드 추가/제거 버튼
            col_a, col_b = st.columns([1, 1])
            with col_a:
                if st.button("➕ 키워드 추가", use_container_width=True, key="add_keyword_btn"):
                    st.session_state.additional_keywords.append({"text": "", "condition": "AND"})
                    st.rerun()
            with col_b:
                if st.button("➖ 키워드 제거", use_container_width=True, key="remove_keyword_btn") and len(st.session_state.additional_keywords) > 0:
                    st.session_state.additional_keywords.pop()
                    st.rerun()
            
            # 추가 키워드 입력 필드들
            for i in range(len(st.session_state.additional_keywords)):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.session_state.additional_keywords[i]["text"] = st.text_input(
                        f"키워드 {i+1}",
                        value=st.session_state.additional_keywords[i]["text"],
                        key=f"crawl_keyword_{i}"
                    )
                with col_b:
                    st.session_state.additional_keywords[i]["condition"] = st.selectbox(
                        "조건",
                        ["AND", "OR"],
                        index=0 if st.session_state.additional_keywords[i]["condition"] == "AND" else 1,
                        key=f"crawl_condition_{i}"
                    )
            
            # 크롤링 옵션 섹션
            st.markdown("##### 크롤링 옵션")
            
            # 플랫폼 선택
            platform_groups = {
                "네이버": ["naver"],
                "유튜브": ["youtube"],
                "구글": ["google"],
                "커뮤니티": ["dcinside", "fmkorea"],
                "부안군청": ["buan"]
            }
            
            selected_groups = st.multiselect(
                "크롤링할 플랫폼 선택",
                options=list(platform_groups.keys()),
                default=list(platform_groups.keys()),
                help="복수 선택 가능. 모든 플랫폼을 동시에 크롤링하거나 원하는 플랫폼만 선택할 수 있습니다."
            )
            
            # 선택된 그룹에서 플랫폼 목록 추출
            platforms = []
            for group in selected_groups:
                platforms.extend(platform_groups[group])
            
            platform_option = ",".join(platforms) if platforms else "all"
            
            # 페이지 수
            pages = st.number_input("페이지/결과 수", min_value=1, max_value=100, value=3, key="crawl_pages")
            
            # 댓글 수
            comments = st.number_input("댓글 수 (유튜브/DC/FMKorea)", min_value=0, max_value=100, value=20, key="crawl_comments")
            
            # 고급 옵션
            with st.expander("고급 옵션"):
                parallel = st.checkbox("병렬 처리", value=True, help="여러 플랫폼을 동시에 크롤링하여 시간을 절약합니다.", key="crawl_parallel")
                no_sentiment = st.checkbox("감성 분석 제외", value=False, help="감성 분석을 제외하고 크롤링만 수행합니다. 속도가 빨라집니다.", key="crawl_no_sentiment")
                ignore_robots = st.checkbox("robots.txt 무시", value=False, help="웹사이트의 robots.txt 정책을 무시하고 크롤링합니다. DCinside와 FMKorea 크롤링에 필요합니다.", key="crawl_ignore_robots")
                max_daily_queries = st.number_input("구글 API 일일 쿼리 제한", min_value=10, max_value=1000, value=100, help="구글 API는 무료 계정에서 일일 100회로 제한됩니다.", key="crawl_google_limit")
            
            # 크롤링 실행 버튼
            st.markdown("---")
            
            # 크롤링 상태 확인
            is_crawling = crawler_status['is_running']
            
            if not is_crawling:
                if st.button("🚀 크롤링 시작", use_container_width=True, type="primary", key="start_crawl_btn"):
                    if not region_keyword.strip():
                        st.error("지역 키워드를 입력해주세요.")
                    else:
                        # 명령어 구성
                        cmd_parts = [
                            "python main.py",
                            f"--keywords {get_keywords_string(region_keyword, st.session_state.additional_keywords)}"
                        ]
                        
                        # 플랫폼 옵션 처리
                        if platforms:
                            platform_option = ",".join(platforms)
                            if platform_option.strip().lower() == "all":
                                st.error("플랫폼을 하나 이상 선택해주세요.")
                                return
                            cmd_parts.append(f"--platform {platform_option}")
                        else:
                            st.error("플랫폼을 하나 이상 선택해주세요.")
                            return
                        
                        # 나머지 옵션 추가
                        cmd_parts.append(f"--max-pages {pages}")
                        cmd_parts.append(f"--max-comments {comments}")
                        
                        # 고급 옵션 추가
                        if parallel:
                            cmd_parts.append("--parallel")
                        if no_sentiment:
                            cmd_parts.append("--no-sentiment")
                        if ignore_robots:
                            cmd_parts.append("--respect-robots")
                        cmd_parts.append(f"--max-daily-queries {max_daily_queries}")
                        
                        # 최종 명령어
                        cmd = " ".join(cmd_parts)
                        
                        # 결과 파일을 저장할 디렉터리 생성
                        os.makedirs("data/raw", exist_ok=True)
                        os.makedirs("data/logs", exist_ok=True)
                        
                        # 명령어 상태 저장
                        crawler_status['command'] = cmd
                        
                        # 로그에 명령어 기록
                        logger.info(f"크롤링 시작: {cmd}")
                        
                        try:
                            # 크롤링 스레드 시작
                            crawling_thread = threading.Thread(
                                target=run_crawler,
                                args=(cmd,)
                            )
                            crawling_thread.daemon = True
                            crawling_thread.start()
                            logger.info(f"크롤링 스레드 시작됨: {crawling_thread.name}")
                        except Exception as e:
                            logger.error(f"크롤링 스레드 시작 실패: {str(e)}")
                            st.error(f"크롤링 스레드 시작 실패: {str(e)}")
                        
                        # 화면 새로고침
                        time.sleep(0.5)  # 스레드가 시작되기를 잠시 기다림
                        st.rerun()
            else:
                if st.button("⏹️ 크롤링 중지", use_container_width=True, type="secondary", key="stop_crawl_btn"):
                    # 실제로 스레드를 중지할 수 없지만 UI에서는 크롤링이 중지된 것처럼 표시
                    crawler_status['is_running'] = False
                    crawler_status['message'] = "크롤링이 중지되었습니다."
                    crawler_status['progress'] = 1.0
                    crawler_status['update_timestamp'] = time.time()
                    save_crawler_status(crawler_status)  # 상태 파일 업데이트
                    logger.info("크롤링 중지 요청")
                    st.warning("크롤링 스레드를 강제로 중단할 수 없습니다. 완료될 때까지 기다려주세요.")
                    st.rerun()
        
        elif dashboard_mode == "데이터 분석":
            st.header("📊 데이터 분석")
            
            # 데이터셋 선택
            datasets = get_available_datasets()
            if not datasets:
                st.warning("사용 가능한 데이터셋이 없습니다.")
                return
            
            # 데이터셋 선택
            selected_dataset = st.selectbox(
                "분석할 데이터셋 선택",
                options=[d['filename'] for d in datasets],
                format_func=lambda x: f"{x} ({datasets[[d['filename'] for d in datasets].index(x)]['count']}개)"
            )
            
            # 감성분석 모델 선택
            data_processor = DataProcessor()
            available_models = data_processor.get_available_models()
            model_combinations = data_processor.model_combinations
            
            st.subheader("모델 선택")
            
            # 개별 모델 선택 (기본 옵션)
            selected_models = st.multiselect(
                "사용할 모델을 선택하세요",
                options=list(available_models.keys()),
                format_func=lambda x: available_models[x],
                default=['kobert', 'kcbert'],
                help="여러 모델을 선택하면 앙상블로 분석됩니다."
            )
            
            # 미리 정의된 조합 선택 (보조 옵션)
            st.markdown("---")
            st.subheader("미리 정의된 모델 조합")
            
            # 조합 설명을 더 가독성 있게 표시
            combinations_info = {
                'light': {
                    'title': '가벼운 조합',
                    'models': ['kobert', 'kcelectra'],
                    'description': '빠른 처리 속도에 최적화된 조합입니다.'
                },
                'balanced': {
                    'title': '균형잡힌 조합',
                    'models': ['kcbert', 'kcelectra', 'kosentencebert'],
                    'description': '속도와 정확도의 균형을 맞춘 조합입니다.'
                },
                'heavy': {
                    'title': '정확도 중심 조합',
                    'models': ['kcbert-large', 'kosentencebert', 'kcelectra'],
                    'description': '높은 정확도를 우선시하는 조합입니다.'
                }
            }
            
            # 조합 선택 UI
            for combo_key, combo_info in combinations_info.items():
                with st.expander(f"📊 {combo_info['title']}"):
                    st.markdown(f"**포함 모델:** {', '.join(combo_info['models'])}")
                    st.markdown(f"*{combo_info['description']}*")
                    if st.button(f"이 조합으로 분석하기", key=f"use_{combo_key}"):
                        try:
                            with st.spinner("데이터셋 분석 중..."):
                                # 선택된 데이터셋의 파일 경로 찾기
                                filepath = next(d['path'] for d in datasets if d['filename'] == selected_dataset)
                                
                                # 선택된 조합으로 분석 실행
                                df = data_processor.analyze_dataset(
                                    input_file=filepath,
                                    models=combo_info['models'],
                                    output_dir="data/processed"
                                )
                                
                                if df is not None and not df.empty:
                                    st.session_state.analysis_data = df
                                    st.session_state.show_results = True
                                    st.session_state.last_dataset = selected_dataset
                                    st.session_state.last_models = combo_info['models']
                                    st.rerun()
                                else:
                                    st.error("분석 결과가 없습니다.")
                        except Exception as e:
                            st.error(f"데이터셋 분석 중 오류 발생: {str(e)}")
                            logger.error(f"데이터셋 분석 중 오류 발생: {str(e)}", exc_info=True)
            
            # 개별 모델 선택 시 분석 버튼
            if selected_models and st.button("선택한 모델로 분석하기"):
                try:
                    with st.spinner("데이터셋 분석 중..."):
                        # 선택된 데이터셋의 파일 경로 찾기
                        filepath = next(d['path'] for d in datasets if d['filename'] == selected_dataset)
                        
                        # 데이터셋 분석 실행 (파일 경로와 모델 목록 전달)
                        df = data_processor.analyze_dataset(
                            input_file=filepath,
                            models=selected_models,
                            output_dir="data/processed"
                        )
                        
                        if df is not None and not df.empty:
                            st.session_state.analysis_data = df
                            st.session_state.show_results = True
                            st.session_state.last_dataset = selected_dataset
                            st.session_state.last_models = selected_models
                            st.rerun()
                        else:
                            st.error("분석 결과가 없습니다.")
                except Exception as e:
                    st.error(f"데이터셋 분석 중 오류 발생: {str(e)}")
                    logger.error(f"데이터셋 분석 중 오류 발생: {str(e)}", exc_info=True)
    
    # 메인 영역 - 크롤링 모드일 때 상태 표시
    if dashboard_mode == "크롤링":
        st.header("🤖 크롤링 상태 모니터링")
        
        # 크롤링 상태 확인
        is_crawling = crawler_status['is_running']
        
        if is_crawling:
            # 진행 상황 표시
            st.subheader("진행 상황")
            progress_value = float(crawler_status['progress'])
            st.progress(progress_value, text=f"{int(progress_value * 100)}% 완료")
            
            # 현재 상태 메시지 표시
            st.info(crawler_status['message'])
            
            # 명령어 표시
            with st.expander("실행 중인 명령어 확인"):
                st.code(crawler_status['command'])
                
            # 자동 새로고침 (5초마다)
            time_since_update = time.time() - crawler_status['update_timestamp']
            if time_since_update > 10:
                st.warning(f"업데이트 없음: {int(time_since_update)}초 동안 상태 업데이트가 없습니다.")
            
            # 메타데이터
            st.caption(f"마지막 업데이트: {datetime.fromtimestamp(crawler_status['update_timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            # 최근 결과 표시
            st.subheader("최근 크롤링 결과")
            results = load_latest_results()
            
            if results:
                # 결과 테이블 표시
                st.write("최근 수집된 데이터셋:")
                result_df = pd.DataFrame([
                    {
                        "플랫폼": r["platform"],
                        "키워드": r["keywords"],
                        "항목 수": r["items"],
                        "수집 시간": r["modified"],
                        "파일명": r["filename"]
                    } for r in results
                ])
                st.dataframe(result_df, use_container_width=True)
                
                # 통계 표시
                stats = get_latest_result_stats()
                if stats:
                    st.subheader("통계 요약")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("총 수집 항목", stats['total'])
                        
                        # 플랫폼별 항목 수
                        st.write("플랫폼별 항목 수:")
                        platform_data = [{"플랫폼": p, "항목 수": c} for p, c in stats['platform_counts'].items()]
                        platform_df = pd.DataFrame(platform_data)
                        st.dataframe(platform_df, use_container_width=True)
                    
                    with col2:
                        # 감성 분포
                        sentiment_data = [{"감성": s, "항목 수": c} for s, c in stats['sentiment_counts'].items()]
                        sentiment_df = pd.DataFrame(sentiment_data)
                        
                        # 차트
                        fig, ax = plt.subplots()
                        bars = ax.bar(sentiment_df['감성'], sentiment_df['항목 수'])
                        
                        # 색상 설정
                        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red', 'unknown': 'lightgray'}
                        for i, bar in enumerate(bars):
                            sentiment = sentiment_df.iloc[i]['감성']
                            bar.set_color(colors.get(sentiment, 'blue'))
                            
                        plt.title("감성 분포")
                        st.pyplot(fig)
                    
                    # 최근 크롤링 시간
                    st.caption(f"최근 크롤링: {results[0]['modified'] if results else '없음'}")
            else:
                st.info("아직 크롤링된 데이터가 없습니다. '크롤링 시작' 버튼을 눌러 데이터 수집을 시작하세요.")
            
            # 데이터 위치 안내
            st.markdown("---")
            st.markdown("##### 수집된 데이터 위치")
            st.code("data/raw/*.json")
            st.caption("수집된 데이터는 위 경로에 JSON 형식으로 저장됩니다. 데이터 분석 모드에서 분석할 수 있습니다.")
    
    # 메인 영역 - 데이터 분석 모드일 때 결과 표시
    if dashboard_mode == "데이터 분석":
        # 감성 분석기 초기화
        if st.session_state.analyzer_option == "Naive Bayes":
            sentiment_analyzer = SentimentAnalyzer()
        elif st.session_state.analyzer_option == "KoBERT":
            sentiment_analyzer = KoBERTSentimentAnalyzer()
        elif st.session_state.analyzer_option == "Ensemble":
            sentiment_analyzer = EnsembleSentimentAnalyzer()
    
        # 기본 시각화 표시 (데이터 유무와 관계없이)
        st.subheader("감성 분포 지도")
        if st.session_state.analysis_data is not None and isinstance(st.session_state.analysis_data, (pd.DataFrame, list)) and len(st.session_state.analysis_data) > 0:
            map_ = create_map(st.session_state.analysis_data)
            if map_ is not None:
                st_folium(map_, width=700, height=500)
            else:
                st.info("지도 데이터가 없습니다.")
        else:
            # 기본 지도 생성 (부안군 중심)
            map_ = folium.Map(location=[35.728, 126.733], zoom_start=10)
            marker_cluster = MarkerCluster().add_to(map_)
            folium.Marker(
                location=[35.728, 126.733],
                popup="부안군",
                icon=folium.Icon(color='blue')
            ).add_to(marker_cluster)
            st_folium(map_, width=700, height=500)
        
        # 감성 분석 샘플 출력
        st.subheader("📋 감성 분석 샘플")
        if st.session_state.get("show_results", False) and st.session_state.get("analysis_data") is not None:
            df = st.session_state.analysis_data
            if not df.empty and all(col in df.columns for col in ['title', 'content', 'sentiment', 'confidence']):
                st.write(df[['title', 'content', 'sentiment', 'confidence']].head())
            else:
                st.warning("필요한 컬럼이 없거나 데이터가 비어있습니다.")
        else:
            st.info("분석할 데이터가 없습니다.")
        
        # CSV 다운로드
        if st.session_state.analysis_data is not None and isinstance(st.session_state.analysis_data, (pd.DataFrame, list)) and len(st.session_state.analysis_data) > 0:
            df = pd.DataFrame(st.session_state.analysis_data) if isinstance(st.session_state.analysis_data, list) else st.session_state.analysis_data
            if not df.empty:
                st.download_button(
                    "📥 CSV 다운로드",
                    df.to_csv(index=False).encode('utf-8'),
                    file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv'
                )
            else:
                st.warning("다운로드할 데이터가 없습니다.")
        
        # 감성 분포 시각화
        st.subheader("감성 분포")
        if st.session_state.analysis_data is not None and isinstance(st.session_state.analysis_data, (pd.DataFrame, list)) and len(st.session_state.analysis_data) > 0:
            df = pd.DataFrame(st.session_state.analysis_data) if isinstance(st.session_state.analysis_data, list) else st.session_state.analysis_data
            if not df.empty and 'sentiment' in df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    sentiment_counts = df['sentiment'].value_counts()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sentiment_counts.plot(kind='bar', ax=ax)
                    plt.title("감성 분포")
                    st.pyplot(fig)
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                    plt.title("감성 분포 (비율)")
                    st.pyplot(fig)
            else:
                st.warning("감성 분석 데이터가 없습니다.")
        else:
            st.info("분석할 데이터가 없습니다.")
        
        # 시계열 트렌드
        st.subheader("시계열 감성 트렌드")
        if st.session_state.analysis_data is not None and len(st.session_state.analysis_data) > 0:
            df = pd.DataFrame(st.session_state.analysis_data)
            
            # 날짜 형식 정규화 및 변환
            valid_dates = []
            for idx, row in df.iterrows():
                try:
                    date_str = row['published_date']
                    if date_str.isdigit() and len(date_str) == 8:
                        df.at[idx, 'date'] = pd.to_datetime(date_str, format='%Y%m%d')
                        valid_dates.append(idx)
                except:
                    pass
            
            # 유효한 날짜만 포함된 데이터프레임 생성
            df_valid = df.loc[valid_dates]
            
            if not df_valid.empty:
                daily_sentiment = df_valid.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
                fig, ax = plt.subplots(figsize=(12, 6))
                daily_sentiment.plot(kind='line', ax=ax)
                plt.title("일별 감성 트렌드")
                st.pyplot(fig)
            else:
                st.warning("시계열 트렌드를 표시할 수 있는 유효한 날짜 데이터가 없습니다.")
        else:
            st.info("분석할 데이터가 없습니다.")
        
        # 워드클라우드
        st.subheader("키워드 워드클라우드")
        if st.session_state.analysis_data is not None and len(st.session_state.analysis_data) > 0:
            df = pd.DataFrame(st.session_state.analysis_data)
            col1, col2 = st.columns(2)
            with col1:
                # 전체 컨텐츠 워드클라우드
                text = ' '.join(df['content'])
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                plt.title("전체 컨텐츠 워드클라우드")
                st.pyplot(fig)
            with col2:
                # 긍정 감성 워드클라우드
                positive_text = ' '.join(df[df['sentiment'] == 'positive']['content'])
                if positive_text.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    plt.title("긍정 감성 워드클라우드")
                    st.pyplot(fig)
                else:
                    st.write("긍정 감성의 텍스트가 없습니다.")
        else:
            st.info("분석할 데이터가 없습니다.")
        
        # GPT 리포트 생성
        st.subheader("정책 제안 리포트")
        if st.session_state.analysis_data is not None and len(st.session_state.analysis_data) > 0:
            try:
                df = pd.DataFrame(st.session_state.analysis_data)
                report_generator = GPTReportGenerator(api_key=os.getenv("OPENAI_API_KEY"))
                report = report_generator.generate_report(df)
                st.text(report)
                
                # PDF 리포트 저장
                pdf_generator = PDFReportGenerator()
                pdf_path = pdf_generator.generate_pdf(report)
                if pdf_path:
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "📄 PDF 리포트 다운로드",
                            f,
                            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime='application/pdf'
                        )
            except Exception as e:
                st.error(f"리포트 생성 중 오류 발생: {str(e)}")
        else:
            st.info("분석할 데이터가 없습니다.")

if __name__ == "__main__":
    main() 