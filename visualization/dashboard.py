import streamlit as st
import folium
from streamlit_folium import st_folium
from utils.data_loader import DataLoader
from core.sentiment_analysis import SentimentAnalyzer
from core.sentiment_analysis_kobert import KoBERTSentimentAnalyzer
from core.sentiment_analysis_ensemble import EnsembleSentimentAnalyzer
from reporting.report_generator_gpt import GPTReportGenerator
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from folium.plugins import MarkerCluster
from reporting.pdf_report_generator import PDFReportGenerator
from crawlers.naver_api_crawler import NaverSearchAPICrawler
from crawlers.youtube import YouTubeCrawler
import os
import glob
from collections import Counter
from dateutil import parser
import json
import random

@st.cache_data
def analyze_sentiment(text, analyzer):
    """텍스트 감성 분석 결과 캐싱"""
    sentiment, confidence = analyzer.predict(text)
    sentiment_label = 'negative' if sentiment == 0 else 'neutral' if sentiment == 1 else 'positive'
    return sentiment_label, confidence

def create_map(data):
    """Folium 지도 생성 및 클러스터링"""
    m = folium.Map(location=[35.728, 126.733], zoom_start=10)
    marker_cluster = MarkerCluster().add_to(m)
    
    for item in data:
        try:
            # 위치 정보가 있는 경우에만 마커 추가
            if 'lat' in item and 'lon' in item:
                folium.Marker(
                    location=[item['lat'], item['lon']],
                    popup=item['content'],
                    icon=folium.Icon(color='blue' if item['sentiment'] == 'positive' else 'red' if item['sentiment'] == 'negative' else 'gray')
                ).add_to(marker_cluster)
            # 위치 정보가 없는 경우 랜덤 오프셋 사용
            else:
                lat = 35.728 + random.uniform(-0.01, 0.01)
                lon = 126.733 + random.uniform(-0.01, 0.01)
                folium.Marker(
                    location=[lat, lon],
                    popup=item['content'],
                    icon=folium.Icon(color='blue' if item['sentiment'] == 'positive' else 'red' if item['sentiment'] == 'negative' else 'gray')
                ).add_to(marker_cluster)
        except Exception as e:
            st.warning(f"마커 추가 중 오류 발생: {str(e)}")
            continue
    
    return m

def get_available_datasets():
    """사용 가능한 데이터셋 목록을 반환"""
    datasets = []
    for file in glob.glob("data/raw/naver_blog_*.json"):
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) >= 4:
            count = parts[2]
            keywords = '_'.join(parts[3:-2])
            datasets.append({
                'filename': filename,
                'count': count,
                'keywords': keywords,
                'path': file
            })
    return sorted(datasets, key=lambda x: x['count'], reverse=True)

def load_selected_dataset(filepath):
    """선택된 데이터셋 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def main():
    """대시보드 메인 함수"""
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
    
    # 사이드바를 상하로 나누기
    with st.sidebar:
        # 상단: 데이터 수집 섹션
        st.header("데이터 수집")
        selected_platform = st.selectbox(
            "플랫폼 선택",
            ["naver_blog", "youtube", "twitter"],
            help="현재는 네이버 블로그만 지원됩니다. 다른 플랫폼은 개발 중입니다."
        )
        
        num_pages = st.number_input("크롤링할 페이지 수", min_value=1, max_value=10000, value=3)
        
        # 키워드 입력 UI
        st.subheader("키워드 설정")
        
        # 지역 키워드 (필수)
        region_keyword = st.text_input("지역 키워드", value=st.session_state.region_keyword)
        if region_keyword.strip():
            st.session_state.region_keyword = region_keyword
        else:
            st.error("지역 키워드는 필수입니다.")
        
        # 추가 키워드 섹션
        st.markdown("##### 추가 키워드")
        
        # 키워드 입력 필드 추가/제거 버튼
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➕ 키워드 추가", use_container_width=True):
                st.session_state.additional_keywords.append({"text": "", "condition": "AND"})
        with col2:
            if st.button("➖ 키워드 제거", use_container_width=True) and len(st.session_state.additional_keywords) > 1:
                st.session_state.additional_keywords.pop()
        
        # 추가 키워드 입력 필드들
        for i in range(len(st.session_state.additional_keywords)):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.session_state.additional_keywords[i]["text"] = st.text_input(
                    f"추가 키워드 {i+1}",
                    value=st.session_state.additional_keywords[i]["text"],
                    key=f"keyword_{i}"
                )
            with col2:
                st.session_state.additional_keywords[i]["condition"] = st.selectbox(
                    "조건",
                    ["AND", "OR"],
                    key=f"condition_{i}"
                )
        
        # 데이터 수집 버튼
        if st.button("🔄 데이터 수집 실행", key="crawl_button"):
            if not st.session_state.region_keyword.strip():
                st.error("❌ 지역 키워드를 입력해주세요.")
            else:
                # 키워드 조합 생성
                keywords = [{"text": st.session_state.region_keyword, "condition": "AND"}]
                keywords.extend([k for k in st.session_state.additional_keywords if k["text"].strip()])
                
                if selected_platform == "naver_blog":
                    crawler = NaverSearchAPICrawler(
                        keywords=keywords,
                        max_pages=num_pages
                    )
                    data = crawler.crawl()
                    if data:
                        st.session_state.analysis_data = data
                        st.session_state.show_results = True
                        st.success(f"✅ 네이버 블로그 데이터 수집 완료! ({len(data)}개)")
                    else:
                        st.error("❌ 데이터 수집에 실패했습니다.")
                elif selected_platform == "youtube":
                    crawler = YouTubeCrawler(keywords=[st.session_state.region_keyword], max_results=30)
                    data = crawler.crawl()
                    if data:
                        st.session_state.analysis_data = data
                        st.session_state.show_results = True
                        st.success("✅ 유튜브 크롤링 완료!")
                    else:
                        st.error("❌ 데이터 수집에 실패했습니다.")
                else:
                    st.warning("⚠️ 해당 플랫폼은 현재 개발 중입니다.")
        
        # 구분선 추가
        st.markdown("---")
        
        # 하단: 분석 설정 섹션
        st.header("분석 설정")
        
        # 데이터셋 선택
        st.subheader("데이터셋 선택")
        datasets = get_available_datasets()
        if datasets:
            dataset_options = {f"{d['count']}개 - {d['keywords']}": d['path'] for d in datasets}
            selected_dataset = st.selectbox(
                "크롤링된 데이터셋",
                options=list(dataset_options.keys())
            )
            selected_filepath = dataset_options[selected_dataset]
            data = load_selected_dataset(selected_filepath)
            
            # 날짜 범위 선택
            st.subheader("날짜 범위 선택")
            dates = sorted(list(set(item['published_date'] for item in data)))
            if dates:
                min_date = datetime.strptime(dates[0], "%Y%m%d")
                max_date = datetime.strptime(dates[-1], "%Y%m%d")
                date_range = st.date_input(
                    "분석 기간 선택",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                # 날짜 필터링
                start_date = datetime.combine(date_range[0], datetime.min.time())
                end_date = datetime.combine(date_range[1], datetime.max.time())
                data = [
                    item for item in data 
                    if start_date <= datetime.strptime(item['published_date'], "%Y%m%d") <= end_date
                ]
        else:
            st.warning("저장된 데이터셋이 없습니다. 데이터를 수집해주세요.")
            data = []
        
        # 감성 분석기 선택
        st.subheader("감성 분석기 선택")
        analyzer_option = st.selectbox("감성 분석기 선택", ["Naive Bayes", "KoBERT", "Ensemble"])
        
        # 분석 실행 버튼
        if st.button("🔍 분석 실행", key="analyze_button"):
            if data:
                st.session_state.analysis_data = data
                st.session_state.analyzer_option = analyzer_option
                st.session_state.show_results = True
                st.success("✅ 분석이 시작되었습니다!")
            else:
                st.error("❌ 분석할 데이터가 없습니다. 데이터셋을 선택해주세요.")
    
    # 메인 페이지에 분석 결과 표시
    st.header("📊 분석 결과")
    
    # 감성 분석기 초기화
    if st.session_state.analyzer_option == "Naive Bayes":
        sentiment_analyzer = SentimentAnalyzer()
    elif st.session_state.analyzer_option == "KoBERT":
        sentiment_analyzer = KoBERTSentimentAnalyzer()
    elif st.session_state.analyzer_option == "Ensemble":
        sentiment_analyzer = EnsembleSentimentAnalyzer()
    
    # 기본 시각화 표시 (데이터 유무와 관계없이)
    st.subheader("감성 분포 지도")
    if st.session_state.analysis_data:
        map_ = create_map(st.session_state.analysis_data)
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
    if st.session_state.analysis_data:
        df = pd.DataFrame(st.session_state.analysis_data)
        st.write(df[['title', 'content', 'sentiment', 'confidence']].head())
    else:
        st.info("분석할 데이터가 없습니다.")
    
    # CSV 다운로드
    if st.session_state.analysis_data:
        df = pd.DataFrame(st.session_state.analysis_data)
        st.download_button(
            "📥 CSV 다운로드",
            df.to_csv(index=False).encode('utf-8'),
            file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )
    
    # 감성 분포 시각화
    st.subheader("감성 분포")
    if st.session_state.analysis_data:
        df = pd.DataFrame(st.session_state.analysis_data)
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
        st.info("분석할 데이터가 없습니다.")
    
    # 시계열 트렌드
    st.subheader("시계열 감성 트렌드")
    if st.session_state.analysis_data:
        df = pd.DataFrame(st.session_state.analysis_data)
        df['date'] = pd.to_datetime(df['published_date'])
        daily_sentiment = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 6))
        daily_sentiment.plot(kind='line', ax=ax)
        plt.title("일별 감성 트렌드")
        st.pyplot(fig)
    else:
        st.info("분석할 데이터가 없습니다.")
    
    # 워드클라우드
    st.subheader("키워드 워드클라우드")
    if st.session_state.analysis_data:
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
    if st.session_state.analysis_data:
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