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
import matplotlib.font_manager as fm
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

# WordCloud 한글 폰트 경로 설정
def get_korean_font_path():
    possible_paths = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
        "/Library/Fonts/AppleGothic.ttf",              # macOS alternative
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
        "C:/Windows/Fonts/malgun.ttf"                  # Windows
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

korean_font_path = get_korean_font_path()

# matplotlib 한글 폰트 설정
def set_matplotlib_font():
    # 시스템에 설치된 폰트 중 나눔고딕 또는 맑은 고딕 찾기
    font_list = [f.name for f in fm.fontManager.ttflist]
    nanum_fonts = [f for f in font_list if 'NanumGothic' in f]
    malgun_fonts = [f for f in font_list if 'Malgun Gothic' in f]
    
    # 폰트 설정
    if nanum_fonts:
        plt.rc('font', family=nanum_fonts[0])
        plt.rcParams['font.family'] = nanum_fonts[0]
    elif malgun_fonts:
        plt.rc('font', family=malgun_fonts[0])
        plt.rcParams['font.family'] = malgun_fonts[0]
    else:
        # 폰트가 없는 경우 경고 메시지 출력
        print("경고: 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        # 기본 폰트 설정
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'AppleGothic', 'DejaVu Sans']
    
    # 마이너스 기호 깨짐 방지
    plt.rc('axes', unicode_minus=False)
    plt.rcParams['axes.unicode_minus'] = False
    
    # 폰트 크기 설정
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # 폰트 설정 확인
    print(f"현재 설정된 폰트: {plt.rcParams['font.family']}")
    print(f"사용 가능한 폰트 목록: {[f.name for f in fm.fontManager.ttflist if 'Gothic' in f.name or 'Nanum' in f.name]}")

# 한글 폰트 설정 적용
set_matplotlib_font()
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
        # 시작 시간이 없으면 현재 시간으로 설정
        if 'start_time' not in status and status['is_running']:
            status['start_time'] = time.time()
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
        crawler_status['message'] = "크롤링 준비 중..."
        crawler_status['progress'] = 0.0
        crawler_status['result'] = ""
        crawler_status['is_running'] = True
        crawler_status['update_timestamp'] = time.time()
        crawler_status['start_time'] = time.time()
        crawler_status['command'] = cmd
        crawler_status['platform_progress'] = {}  # 플랫폼별 진행률 저장
        crawler_status['total_items'] = 0  # 전체 항목 수
        crawler_status['processed_items'] = 0  # 처리된 항목 수
        save_crawler_status(crawler_status)
        
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
                
            # 플랫폼별 진행률 초기화
            for platform in platforms:
                crawler_status['platform_progress'][platform] = 0.0
            
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
                            crawler_status['update_timestamp'] = time.time()
                            save_crawler_status(crawler_status)
                            logger.info(status_msg)
                            break
                
                # 항목 수 업데이트
                if "수집된 항목:" in line:
                    try:
                        items_count = int(line.split("수집된 항목:")[1].strip())
                        crawler_status['total_items'] = max(crawler_status['total_items'], items_count)
                        if current_platform:
                            crawler_status['platform_progress'][current_platform] = min(0.99, items_count / 100)  # 임시 진행률
                        save_crawler_status(crawler_status)
                    except:
                        pass
                
                # 플랫폼별 완료 확인
                if "크롤링 완료" in line:
                    for platform in platforms:
                        if platform.upper() in line:
                            crawler_status['platform_progress'][platform] = 1.0
                            crawler_status['processed_items'] += 1
                            # 전체 진행률 계산
                            total_progress = sum(crawler_status['platform_progress'].values()) / len(platforms)
                            crawler_status['progress'] = total_progress
                            crawler_status['update_timestamp'] = time.time()
                            save_crawler_status(crawler_status)
                            break
                
                # 최종 결과 확인
                if "통합 결과 저장 경로" in line:
                    result_path = line.split("통합 결과 저장 경로:")[1].strip()
                    crawler_status['result'] = f"✅ 크롤링 완료! 결과 저장 경로: {result_path}"
                    crawler_status['progress'] = 1.0
                    crawler_status['update_timestamp'] = time.time()
                    save_crawler_status(crawler_status)
                    logger.info(f"크롤링 완료, 결과 저장 경로: {result_path}")
            
            # 프로세스가 완료될 때까지 기다림
            return_code = process.wait()
            logger.info(f"크롤링 프로세스 종료, 리턴 코드: {return_code}")
            
            # 크롤링 완료
            crawler_status['progress'] = 1.0
            crawler_status['message'] = "✅ 크롤링 완료!"
            crawler_status['is_running'] = False
            crawler_status['update_timestamp'] = time.time()
            save_crawler_status(crawler_status)
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
        save_crawler_status(crawler_status)
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
        # 모든 플랫폼의 최근 결과 파일 검색
        platforms = {
            "naver_news": "네이버 뉴스",
            "naver_blog": "네이버 블로그",
            "naver_cafearticle": "네이버 카페",
            "youtube": "유튜브",
            "google": "구글",
            "dcinside": "디시인사이드",
            "fmkorea": "에펨코리아",
            "buan": "부안군청"
        }
        
        all_results = []
        platform_counts = {}
        latest_files = {}
        
        # 각 플랫폼별 최근 파일 검색
        for platform in platforms.keys():
            files = glob.glob(f"data/raw/{platform}*.json")
            if files:
                # 가장 최근 파일 선택
                latest_file = max(files, key=os.path.getmtime)
                latest_files[platform] = latest_file
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_results.extend(data)
                            platform_counts[platforms[platform]] = len(data)
                            logger.info(f"{platforms[platform]}: {len(data)}개 항목 로드됨")
                except Exception as e:
                    logger.error(f"{platform} 결과 파일 로드 중 오류: {str(e)}")
        
        if not all_results:
            logger.warning("수집된 데이터가 없습니다.")
            return None
        
        # 감성별 항목 수
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0, 'unknown': 0}
        for item in all_results:
            sentiment = item.get('sentiment')
            if sentiment is None:
                sentiment_counts['unknown'] += 1
            elif sentiment == 0 or sentiment == 'negative':
                sentiment_counts['negative'] += 1
            elif sentiment == 1 or sentiment == 'neutral':
                sentiment_counts['neutral'] += 1
            elif sentiment == 2 or sentiment == 'positive':
                sentiment_counts['positive'] += 1
            else:
                sentiment_counts['unknown'] += 1
        
        # 날짜별 항목 수
        date_counts = {}
        for item in all_results:
            date = item.get('published_date', '')
            if date and len(date) >= 8:  # 'YYYYMMDD' 형식
                try:
                    date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                except:
                    date = 'unknown'
            else:
                date = 'unknown'
            date_counts[date] = date_counts.get(date, 0) + 1
        
        # 결과 로깅
        logger.info(f"총 {len(all_results)}개 항목 처리됨")
        logger.info(f"플랫폼별 항목 수: {platform_counts}")
        logger.info(f"감성별 항목 수: {sentiment_counts}")
        
        return {
            'total': len(all_results),
            'platform_counts': platform_counts,
            'sentiment_counts': sentiment_counts,
            'date_counts': date_counts,
            'latest_files': latest_files
        }
    except Exception as e:
        logger.error(f"결과 통계 계산 중 오류 발생: {str(e)}", exc_info=True)
        return None

@st.cache_data
def analyze_sentiment(text, analyzer):
    """텍스트 감성 분석 결과 캐싱"""
    sentiment, confidence = analyzer.predict(text)
    sentiment_label = 'negative' if sentiment == 0 else 'neutral' if sentiment == 1 else 'positive'
    return sentiment_label, confidence

def clear_analysis_cache():
    """분석 관련 캐시와 세션 상태 초기화"""
    try:
        # 세션 상태 초기화
        for key in list(st.session_state.keys()):
            if key.startswith('analysis_') or key in ['show_results', 'last_dataset', 'last_models']:
                del st.session_state[key]
        
        # Streamlit 캐시 초기화
        analyze_sentiment.clear()
        
        logger.info("모든 분석 캐시가 초기화되었습니다.")
        return True
    except Exception as e:
        logger.error(f"캐시 초기화 중 오류 발생: {str(e)}")
        return False

def get_available_datasets():
    """사용 가능한 데이터셋 목록을 반환"""
    # 분석 결과 파일 목록
    analysis_results = []
    processed_files = glob.glob("data/processed/sentiment_analysis_*.json")
    
    for json_file in processed_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # CSV 파일이 존재하는지 확인
            csv_file = metadata['output_files']['csv']
            if not os.path.exists(csv_file):
                continue
            
            # 분석 결과 정보 구성
            result_info = {
                'filename': os.path.basename(csv_file),
                'analysis_time': metadata['timestamp'],
                'models': metadata['models'],
                'item_count': metadata['item_count'],
                'sentiment_distribution': metadata['sentiment_distribution'],
                'csv_file': csv_file,
                'json_file': json_file,
                'input_file': metadata['input_file']
            }
            analysis_results.append(result_info)
            
        except Exception as e:
            logger.error(f"분석 결과 파일 처리 중 오류: {str(e)}")
            continue
    
    # 분석 시간 기준으로 정렬 (최신순)
    analysis_results.sort(key=lambda x: x['analysis_time'], reverse=True)
    
    # 원본 데이터셋 목록
    raw_datasets = []
    for platform in ['naver', 'youtube', 'google', 'dcinside', 'fmkorea', 'buan']:
        for file in glob.glob(f"data/raw/{platform}_*.json"):
            try:
                filename = os.path.basename(file)
                parts = filename.split('_')
                
                if len(parts) >= 3:
                    if platform == 'youtube':
                        count = parts[1]
                        keywords = '_'.join(parts[2:-2])
                    else:
                        count = parts[1]
                        keywords = '_'.join(parts[2:-2])
                        
                    # 이미 분석된 파일인지 확인
                    is_analyzed = any(r['input_file'] == file for r in analysis_results)
                    
                    raw_datasets.append({
                        'filename': filename,
                        'platform': platform,
                        'path': file,
                        'count': count,
                        'keywords': keywords,
                        'modified_time': datetime.fromtimestamp(os.path.getmtime(file)).strftime("%Y-%m-%d %H:%M:%S"),
                        'is_analyzed': is_analyzed
                    })
            except:
                continue
    
    # 수정 시간 기준으로 정렬 (최신순)
    raw_datasets.sort(key=lambda x: x['modified_time'], reverse=True)
    
    return {
        'analysis_results': analysis_results,
        'raw_datasets': raw_datasets
    }

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

def estimate_processing_time(data_size, models_count):
    """데이터 크기와 모델 수에 따른 예상 처리 시간 계산"""
    # 각 모델별 평균 처리 시간 (초)
    model_processing_times = {
        'kobert': 0.5,
        'kcelectra-base-v2022': 0.4,
        'kcelectra': 0.4,
        'kcbert-large': 1.2,
        'kosentencebert': 0.8
    }
    
    # 기본 처리 시간 (데이터 로딩, 전처리 등)
    base_time = 2.0
    
    # 모델별 처리 시간 합산
    total_model_time = sum(model_processing_times.get(model, 0.5) for model in models_count)
    
    # 전체 예상 시간 계산 (초)
    estimated_time = (base_time + total_model_time) * data_size
    
    return estimated_time

def format_time(seconds):
    """초를 읽기 쉬운 시간 형식으로 변환"""
    if seconds < 60:
        return f"{int(seconds)}초"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}분 {remaining_seconds}초"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}시간 {minutes}분"

def update_analysis_progress(progress_bar, status_text, current, total, start_time):
    """분석 진행 상황 업데이트"""
    progress = current / total
    progress_bar.progress(progress)
    
    # 경과 시간 계산
    elapsed_time = time.time() - start_time
    # 남은 시간 예측
    if current > 0:
        estimated_total = (elapsed_time / current) * total
        remaining_time = estimated_total - elapsed_time
        status_text.text(f"진행률: {int(progress * 100)}% ({current}/{total} 항목)\n"
                        f"경과 시간: {format_time(elapsed_time)}\n"
                        f"예상 남은 시간: {format_time(remaining_time)}")
    else:
        status_text.text(f"진행률: 0% (0/{total} 항목)\n"
                        f"경과 시간: {format_time(elapsed_time)}")

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
    if 'analysis_start_time' not in st.session_state:
        st.session_state.analysis_start_time = None
    if 'analysis_progress' not in st.session_state:
        st.session_state.analysis_progress = 0
    if 'analysis_total' not in st.session_state:
        st.session_state.analysis_total = 0
    
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
            with st.expander("고급 옵션", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    browser_type = st.selectbox(
                        "브라우저 선택",
                        ["chrome", "firefox"],
                        index=0,
                        help="크롤링에 사용할 브라우저를 선택합니다."
                    )
                    max_daily_queries = st.number_input(
                        "구글 API 일일 최대 쿼리 수",
                        min_value=10,
                        max_value=1000,
                        value=100,
                        step=10,
                        help="구글 API의 일일 최대 쿼리 수를 설정합니다."
                    )
                with col2:
                    respect_robots = st.checkbox(
                        "robots.txt 정책 준수",
                        value=False,
                        help="웹사이트의 robots.txt 정책을 준수하며 크롤링합니다."
                    )
                    perform_sentiment = st.checkbox(
                        "감성 분석 수행",
                        value=False,
                        help="크롤링 중 감성 분석을 수행합니다."
                    )
            
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
                        if browser_type != "chrome":
                            cmd_parts.append(f"--browser {browser_type}")
                        if respect_robots:
                            cmd_parts.append("--respect-robots")
                        if not perform_sentiment:
                            cmd_parts.append("--no-sentiment")
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
            
            # 사이드바에 원본 데이터 선택 UI 배치
            with st.sidebar:
                st.header("원본 데이터 선택")
                
                # 원본 데이터셋 목록
                datasets = get_available_datasets()
                raw_datasets = datasets['raw_datasets']
                
                if not raw_datasets:
                    st.warning("사용 가능한 원본 데이터셋이 없습니다.")
                else:
                    # 원본 데이터셋 선택 옵션 구성
                    dataset_options = []
                    for dataset in raw_datasets:
                        if dataset['is_analyzed']:
                            label = f"📊 {dataset['filename']} ({dataset['count']}개) - 이미 분석됨"
                        else:
                            label = f"📄 {dataset['filename']} ({dataset['count']}개) - 분석 전"
                        dataset_options.append((label, dataset))
                    
                    # 원본 데이터셋 선택
                    selected_label = st.selectbox(
                        "분석할 데이터셋 선택",
                        options=[opt[0] for opt in dataset_options],
                        help="📊는 이미 분석된 데이터셋, 📄는 분석 전 데이터셋입니다."
                    )
                    
                    # 선택된 데이터셋 정보 가져오기
                    selected_dataset = next(opt[1] for opt in dataset_options if opt[0] == selected_label)
                    
                    # 이미 분석된 데이터셋인 경우 경고
                    if selected_dataset['is_analyzed']:
                        st.warning("이 데이터셋은 이미 분석되었습니다. 분석 결과를 확인하세요.")
                    
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
                default=['kobert', 'kcelectra-base-v2022'],
                help="여러 모델을 선택하면 앙상블로 분석됩니다."
            )
            
            # 미리 정의된 조합 선택 (보조 옵션)
            st.markdown("---")
            st.subheader("미리 정의된 모델 조합")
            
            # 조합 설명을 더 가독성 있게 표시
            combinations_info = {
                'light': {
                    'title': '가벼운 조합',
                    'models': ['kobert', 'kcelectra-base-v2022'],
                    'description': '빠른 처리 속도에 최적화된 조합입니다.'
                },
                'balanced': {
                    'title': '균형잡힌 조합',
                    'models': ['kcelectra-base-v2022', 'kcelectra', 'kosentencebert'],
                    'description': '속도와 정확도의 균형을 맞춘 조합입니다.'
                },
                'heavy': {
                    'title': '정확도 중심 조합',
                    'models': ['kcbert-large', 'kosentencebert', 'kcelectra-base-v2022'],
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
                                filepath = selected_dataset['path']
                                
                                # 선택된 조합으로 분석 실행
                                df = data_processor.analyze_dataset(
                                    input_file=filepath,
                                    models=combo_info['models'],
                                    output_dir="data/processed"
                                )
                                
                                if df is not None and not df.empty:
                                    st.session_state.analysis_data = df
                                    st.session_state.show_results = True
                                    st.session_state.last_dataset = selected_dataset['filename']
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
                    # 선택된 데이터셋의 파일 경로 찾기
                    filepath = selected_dataset['path']
                    
                    # 데이터 크기 확인
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data_size = len(data)
                    
                    # 예상 처리 시간 계산
                    estimated_time = estimate_processing_time(data_size, selected_models)
                    
                    # 진행 상황 표시를 위한 컨테이너 생성
                    progress_container = st.empty()
                    status_container = st.empty()
                    progress_bar = progress_container.progress(0)
                    status_text = status_container.text("분석 준비 중...")
                    
                    # 분석 시작 시간 기록
                    st.session_state.analysis_start_time = time.time()
                    st.session_state.analysis_progress = 0
                    st.session_state.analysis_total = data_size
                    
                    # 예상 시간 표시
                    st.info(f"예상 처리 시간: {format_time(estimated_time)} (데이터 {data_size}개, 모델 {len(selected_models)}개)")
                    
                    with st.spinner("데이터셋 분석 중..."):
                        # 데이터셋 분석 실행
                        df = data_processor.analyze_dataset(
                            input_file=filepath,
                            models=selected_models,
                            output_dir="data/processed",
                            progress_callback=lambda current: update_analysis_progress(
                                progress_bar,
                                status_text,
                                current,
                                data_size,
                                st.session_state.analysis_start_time
                            )
                        )
                    
                    if df is not None and not df.empty:
                        st.session_state.analysis_data = df
                        st.session_state.show_results = True
                        st.session_state.last_dataset = selected_dataset['filename']
                        st.session_state.last_models = selected_models
                        
                        # 진행 상황 컨테이너 제거
                        progress_container.empty()
                        status_container.empty()
                        
                        st.success("분석이 완료되었습니다!")
                    st.rerun()
                except Exception as e:
                    st.error(f"데이터셋 분석 중 오류 발생: {str(e)}")
                    logger.error(f"데이터셋 분석 중 오류 발생: {str(e)}", exc_info=True)

                # 캐시 초기화 버튼 위치 변경 및 메시지 개선
            if st.sidebar.button("분석 캐시 초기화", type="primary"):
                if clear_analysis_cache():
                    st.sidebar.success("✅ 모든 분석 캐시가 초기화되었습니다. 페이지를 새로고침하세요.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.sidebar.error("❌ 캐시 초기화 중 오류가 발생했습니다.")
    
    # 메인 영역 - 크롤링 모드일 때 상태 표시
    if dashboard_mode == "크롤링":
        # 크롤링 상태 컨테이너 생성
        status_container = st.empty()
        
        # 크롤링 중이 아닐 때는 최근 결과 표시
        if not crawler_status['is_running']:
            with status_container.container():
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
        else:
            # 크롤링 중일 때는 상태 정보 표시
            with status_container.container():
                st.subheader("크롤링 상태")
                
                # 상태 정보 표시
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "상태",
                        "실행 중",
                        delta=None
                    )
                with col2:
                    if crawler_status.get('start_time'):
                        st.metric(
                            "시작 시간",
                            datetime.fromtimestamp(crawler_status['start_time']).strftime("%Y-%m-%d %H:%M:%S"),
                            delta=None
                        )
                
                # 플랫폼별 진행 상황
                st.subheader("플랫폼별 진행 상황")
                platform_progress = crawler_status.get('platform_progress', {})
                
                for platform, progress in platform_progress.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"{platform.upper()}")
                        st.progress(progress)
                        st.text(f"{progress*100:.1f}%")
                    with col2:
                        if crawler_status.get('platform_times', {}).get(platform, {}).get('start'):
                            start_time = datetime.strptime(
                                crawler_status['platform_times'][platform]['start'],
                                "%Y-%m-%d %H:%M:%S"
                            )
                            if crawler_status['platform_times'][platform].get('end'):
                                end_time = datetime.strptime(
                                    crawler_status['platform_times'][platform]['end'],
                                    "%Y-%m-%d %H:%M:%S"
                                )
                                duration = end_time - start_time
                                st.text(f"소요 시간: {duration}")
                            else:
                                elapsed = datetime.now() - start_time
                                st.text(f"경과 시간: {elapsed}")
                
                # 크롤링 결과 요약
                st.subheader("크롤링 결과 요약")
                if crawler_status.get('total_items', 0) > 0:
                    st.metric("총 수집 항목", crawler_status['total_items'])
                    st.metric("처리된 항목", crawler_status.get('processed_items', 0))
                
                # 오류 메시지가 있는 경우 표시
                if crawler_status.get('error'):
                    st.error(f"오류 발생: {crawler_status['error']}")
                
                # 자동 새로고침을 위한 대기
                time.sleep(1)
                st.experimental_rerun()
    elif dashboard_mode == "데이터 분석":
        # 메인 화면에 분석 결과 표시
        # 분석 결과 파일 목록
        analysis_results = datasets['analysis_results']
        
        if not analysis_results:
            st.info("분석된 결과가 없습니다. 사이드바에서 새로운 분석을 시작하세요.")
        else:
            # 분석 결과 선택 옵션 구성
            result_options = []
            for result in analysis_results:
                sentiment_dist = result['sentiment_distribution']
                label = (f"📊 {result['filename']} ({result['item_count']}개) - "
                        f"분석: {result['analysis_time']} - "
                        f"감성: {sentiment_dist.get('positive', 'N/A')} 긍정")
                result_options.append((label, result))
            
            # 분석 결과 선택
            selected_label = st.selectbox(
                "분석 결과 선택",
                options=[opt[0] for opt in result_options],
                help="최신 분석 결과가 상단에 표시됩니다."
            )
            
            # 선택된 결과 정보 가져오기
            selected_result = next(opt[1] for opt in result_options if opt[0] == selected_label)
            
            # 분석 결과 데이터 로드
            try:
                df = pd.read_csv(selected_result['csv_file'])
                
                # 감성 값 변환 (숫자 -> 문자열)
                sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                df['sentiment'] = df['sentiment'].map(sentiment_map)
                
                # 신뢰도 백분율로 변환
                df['confidence'] = df['confidence'].apply(lambda x: f"{float(x)*100:.1f}%")
                
                st.session_state.analysis_data = df
                st.session_state.show_results = True
                
                # 감성 분포 계산
                sentiment_counts = df['sentiment'].value_counts()
                total = len(df)
                sentiment_distribution = {
                    'positive': f"{sentiment_counts.get('positive', 0) / total * 100:.1f}%",
                    'neutral': f"{sentiment_counts.get('neutral', 0) / total * 100:.1f}%",
                    'negative': f"{sentiment_counts.get('negative', 0) / total * 100:.1f}%"
                }
                
                # 분석 정보 업데이트
                with st.expander("📋 분석 정보", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**분석 시간:** {selected_result['analysis_time']}")
                        st.markdown(f"**사용 모델:** {', '.join(selected_result['models'])}")
                    with col2:
                        st.markdown(f"**항목 수:** {len(df)}개")
                        st.markdown("**감성 분포:**")
                        st.metric("긍정", sentiment_distribution['positive'])
                        st.metric("중립", sentiment_distribution['neutral'])
                        st.metric("부정", sentiment_distribution['negative'])
                
            except Exception as e:
                st.error(f"분석 결과 로드 중 오류 발생: {str(e)}")
                logger.error(f"분석 결과 로드 중 오류 발생: {str(e)}", exc_info=True)
            
            if st.session_state.analysis_data is not None and isinstance(st.session_state.analysis_data, pd.DataFrame) and not st.session_state.analysis_data.empty:
                df = st.session_state.analysis_data
                
                st.markdown("### 📊 문장 감성 분포", help="이 데이터는 지역별 감성 분석 결과를 지도에 시각화하는 데 활용할 수 있습니다. 연속지적도와 연계하여 특정 지역의 긍정/부정 피드백을 지도상에 표시하면 지역 개발 및 정책 수립에 더욱 효과적인 인사이트를 제공할 수 있습니다.")
                
                # 지역 키워드 관련 문장 필터링 및 정렬
                region_keywords = ['부안', '변산', '내소', '채석강', '고사포', '격포', '위도', '계화', '줄포']
                
                # 지역 키워드가 포함된 문장 찾기
                df['has_region_keyword'] = df['content'].apply(
                    lambda x: any(keyword in x for keyword in region_keywords)
                )
                
                # 신뢰도가 높은 순으로 정렬 (confidence 컬럼에서 % 제거하고 float로 변환)
                df['confidence_value'] = df['confidence'].str.rstrip('%').astype(float)
                
                # 지역 키워드가 포함된 문장 중 신뢰도가 높은 순으로 정렬
                region_df = df[df['has_region_keyword']].sort_values(
                    by=['confidence_value', 'has_region_keyword'],
                    ascending=[False, False]
                ).head(50)
                
                # 감성별로 색상 지정
                sentiment_colors = {
                    'positive': '🟢',  # 긍정: 초록색
                    'neutral': '⚪',   # 중립: 흰색
                    'negative': '🔴'   # 부정: 빨간색
                }
                
                # 감성별 통계
                sentiment_stats = region_df['sentiment'].value_counts()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("긍정", f"{sentiment_stats.get('positive', 0)}개")
                with col2:
                    st.metric("중립", f"{sentiment_stats.get('neutral', 0)}개")
                with col3:
                    st.metric("부정", f"{sentiment_stats.get('negative', 0)}개")
                
                # 감성별로 데이터 분리
                positive_df = region_df[region_df['sentiment'] == 'positive']
                negative_df = region_df[region_df['sentiment'] == 'negative']
                neutral_df = region_df[region_df['sentiment'] == 'neutral']
                
                # 긍정 문장 표시 (상위 5개)
                if not positive_df.empty:
                    st.markdown("### 🟢 긍정 문장")
                    # 상위 5개 문장만 표시
                    for _, row in positive_df.head(5).iterrows():
                        with st.expander(f"**{row['title']}** (신뢰도: {row['confidence']})"):
                            if 'platform' in row:
                                st.caption(f"출처: {row['platform']}")
                            if 'published_date' in row:
                                st.caption(f"작성일: {row['published_date']}")
                            
                            # 지역 키워드가 있는 부분 찾기
                            content = row['content']
                            for keyword in region_keywords:
                                if keyword in content:
                                    # 키워드 주변 텍스트 추출 (앞뒤 50자)
                                    idx = content.find(keyword)
                                    start = max(0, idx - 50)
                                    end = min(len(content), idx + len(keyword) + 50)
                                    highlight = content[start:end]
                                    if start > 0:
                                        highlight = "..." + highlight
                                    if end < len(content):
                                        highlight = highlight + "..."
                                    st.markdown(highlight)
                                    break
                            
                            st.markdown("**전체 내용:**")
                            st.markdown(content)
                    
                    # 5개 초과하는 경우 더 보기 버튼 표시
                    if len(positive_df) > 5:
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button(f"🟢 긍정 문장 더 보기 ({len(positive_df)-5}개 더 있음)", key="more_positive"):
                                st.session_state.show_all_positive = True
                                st.rerun()
                        with col2:
                            if st.session_state.get('show_all_positive', False):
                                if st.button("닫기", key="close_positive_top"):
                                    st.session_state.show_all_positive = False
                                    st.rerun()
                    
                    # 긍정 문장 더 보기 섹션
                    if st.session_state.get('show_all_positive', False):
                        st.markdown("#### 🟢 긍정 문장 전체 보기")
                        for _, row in positive_df.iterrows():
                            with st.expander(f"**{row['title']}** (신뢰도: {row['confidence']})"):
                                if 'platform' in row:
                                    st.caption(f"출처: {row['platform']}")
                                if 'published_date' in row:
                                    st.caption(f"작성일: {row['published_date']}")
                                st.markdown(row['content'])
                        if st.button("닫기", key="close_positive_bottom"):
                            st.session_state.show_all_positive = False
                            st.rerun()
                
                # 부정 문장 표시 (상위 5개)
                if not negative_df.empty:
                    st.markdown("### 🔴 부정 문장")
                    # 상위 5개 문장만 표시
                    for _, row in negative_df.head(5).iterrows():
                        with st.expander(f"**{row['title']}** (신뢰도: {row['confidence']})"):
                            if 'platform' in row:
                                st.caption(f"출처: {row['platform']}")
                            if 'published_date' in row:
                                st.caption(f"작성일: {row['published_date']}")
                            
                            # 지역 키워드가 있는 부분 찾기
                            content = row['content']
                            for keyword in region_keywords:
                                if keyword in content:
                                    # 키워드 주변 텍스트 추출 (앞뒤 50자)
                                    idx = content.find(keyword)
                                    start = max(0, idx - 50)
                                    end = min(len(content), idx + len(keyword) + 50)
                                    highlight = content[start:end]
                                    if start > 0:
                                        highlight = "..." + highlight
                                    if end < len(content):
                                        highlight = highlight + "..."
                                    st.markdown(highlight)
                                    break
                            
                            st.markdown("**전체 내용:**")
                            st.markdown(content)
                    
                    # 5개 초과하는 경우 더 보기 버튼 표시
                    if len(negative_df) > 5:
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button(f"🔴 부정 문장 더 보기 ({len(negative_df)-5}개 더 있음)", key="more_negative"):
                                st.session_state.show_all_negative = True
                                st.rerun()
                        with col2:
                            if st.session_state.get('show_all_negative', False):
                                if st.button("닫기", key="close_negative_top"):
                                    st.session_state.show_all_negative = False
                                    st.rerun()
                    
                    # 부정 문장 더 보기 섹션
                    if st.session_state.get('show_all_negative', False):
                        st.markdown("#### 🔴 부정 문장 전체 보기")
                        for _, row in negative_df.iterrows():
                            with st.expander(f"**{row['title']}** (신뢰도: {row['confidence']})"):
                                if 'platform' in row:
                                    st.caption(f"출처: {row['platform']}")
                                if 'published_date' in row:
                                    st.caption(f"작성일: {row['published_date']}")
                                st.markdown(row['content'])
                        if st.button("닫기", key="close_negative_bottom"):
                            st.session_state.show_all_negative = False
                            st.rerun()
                
                # 중립 문장 표시 (상위 5개)
                if not neutral_df.empty:
                    st.markdown("### ⚪ 중립 문장")
                    # 상위 5개 문장만 표시
                    for _, row in neutral_df.head(5).iterrows():
                        with st.expander(f"**{row['title']}** (신뢰도: {row['confidence']})"):
                            if 'platform' in row:
                                st.caption(f"출처: {row['platform']}")
                            if 'published_date' in row:
                                st.caption(f"작성일: {row['published_date']}")
                            
                            # 지역 키워드가 있는 부분 찾기
                            content = row['content']
                            for keyword in region_keywords:
                                if keyword in content:
                                    # 키워드 주변 텍스트 추출 (앞뒤 50자)
                                    idx = content.find(keyword)
                                    start = max(0, idx - 50)
                                    end = min(len(content), idx + len(keyword) + 50)
                                    highlight = content[start:end]
                                    if start > 0:
                                        highlight = "..." + highlight
                                    if end < len(content):
                                        highlight = highlight + "..."
                                    st.markdown(highlight)
                                    break
                            
                            st.markdown("**전체 내용:**")
                            st.markdown(content)
                    
                    # 5개 초과하는 경우 더 보기 버튼 표시
                    if len(neutral_df) > 5:
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button(f"⚪ 중립 문장 더 보기 ({len(neutral_df)-5}개 더 있음)", key="more_neutral"):
                                st.session_state.show_all_neutral = True
                                st.rerun()
                        with col2:
                            if st.session_state.get('show_all_neutral', False):
                                if st.button("닫기", key="close_neutral_top"):
                                    st.session_state.show_all_neutral = False
                                    st.rerun()
                    
                    # 중립 문장 더 보기 섹션
                    if st.session_state.get('show_all_neutral', False):
                        st.markdown("#### ⚪ 중립 문장 전체 보기")
                        for _, row in neutral_df.iterrows():
                            with st.expander(f"**{row['title']}** (신뢰도: {row['confidence']})"):
                                if 'platform' in row:
                                    st.caption(f"출처: {row['platform']}")
                                if 'published_date' in row:
                                    st.caption(f"작성일: {row['published_date']}")
                                st.markdown(row['content'])
                        if st.button("닫기", key="close_neutral_bottom"):
                            st.session_state.show_all_neutral = False
                            st.rerun()
                
                # 지역 키워드가 없는 경우 안내
                if len(region_df) == 0:
                    st.info("지역 키워드가 포함된 문장을 찾을 수 없습니다.")
                
                # CSV 다운로드 버튼 추가
                st.markdown("### 📥 데이터 다운로드")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "CSV 파일 다운로드",
                    csv,
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv',
                    help="분석된 전체 데이터를 CSV 파일로 다운로드합니다."
                )
        
        # 감성 분포 시각화
                st.markdown("### 📊 감성 분포")
                if st.session_state.analysis_data is not None and isinstance(st.session_state.analysis_data, pd.DataFrame) and not st.session_state.analysis_data.empty:
                    df = st.session_state.analysis_data
                    if 'sentiment' in df.columns:
                        col1, col2 = st.columns(2)
                        with col1:
                            # 막대 그래프
                            sentiment_counts = df['sentiment'].value_counts()
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.bar(sentiment_counts.index, sentiment_counts.values)
                            
                            # 색상 설정
                            colors = {
                                'positive': '#2ecc71',  # 긍정: 초록색
                                'neutral': '#95a5a6',   # 중립: 회색
                                'negative': '#e74c3c'   # 부정: 빨간색
                            }
                            for bar, sentiment in zip(bars, sentiment_counts.index):
                                bar.set_color(colors.get(sentiment, '#3498db'))
                            
                            plt.title("감성 분포")
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                            
                        with col2:
                            # 파이 차트
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sentiment_counts.plot(
                                kind='pie',
                                autopct='%1.1f%%',
                                ax=ax,
                                colors=[colors.get(s, '#3498db') for s in sentiment_counts.index]
                            )
                            plt.title("감성 분포 (비율)")
                            st.pyplot(fig)
                    else:
                        st.info("감성 분석 데이터가 없습니다.")
                else:
                    st.info("분석할 데이터가 없습니다.")
        
        # 시계열 트렌드
                st.markdown("### 📈 시계열 감성 트렌드")
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
                st.markdown("### ☁️ 키워드 워드클라우드")
                if st.session_state.analysis_data is not None and len(st.session_state.analysis_data) > 0:
                    df = pd.DataFrame(st.session_state.analysis_data)
                    col1, col2 = st.columns(2)
                    with col1:
                        # 전체 컨텐츠 워드클라우드
                        text = ' '.join(df['content'])
                        if text.strip():
                            wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=korean_font_path).generate(text)
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            plt.title("전체 컨텐츠 워드클라우드")
                            st.pyplot(fig)
                        else:
                            st.info("워드클라우드를 생성할 텍스트가 없습니다.")
                    with col2:
                        # 긍정 감성 워드클라우드
                        positive_text = ' '.join(df[df['sentiment'] == 'positive']['content'])
                        if positive_text.strip():
                            wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=korean_font_path).generate(positive_text)
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            plt.title("긍정 감성 워드클라우드")
                            st.pyplot(fig)
                        else:
                            st.info("긍정 감성의 텍스트가 없습니다.")
                else:
                    st.info("분석할 데이터가 없습니다.")
        
        # GPT 리포트 생성
                st.markdown("### 📝 정책 제안 리포트")
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
    