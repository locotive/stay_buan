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
            folium.Marker(
                location=[item['lat'], item['lon']],
                popup=item['content'],
                icon=folium.Icon(color='blue' if item['sentiment'] == 'positive' else 'red' if item['sentiment'] == 'negative' else 'gray')
            ).add_to(marker_cluster)
        except TypeError:
            continue
    
    return m

def plot_time_series(data):
    """시계열 감성 트렌드 플롯"""
    df = pd.DataFrame(data)
    
    if 'date' not in df.columns:
        st.write("날짜 정보가 없습니다.")
        return
    
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    sentiment_counts = df.resample('D').sentiment.value_counts().unstack().fillna(0)
    sentiment_counts.plot(kind='line', figsize=(10, 5))
    plt.title("시계열 감성 트렌드")
    plt.xlabel("날짜")
    plt.ylabel("감성 수")
    plt.legend(["부정", "중립", "긍정"])
    st.pyplot(plt)

def parse_user_query(query):
    """사용자 질의 파싱"""
    filters = {}
    if "부정" in query:
        filters['sentiment'] = 'negative'
    if "병원" in query:
        filters['keyword'] = '병원'
    # 추가적인 필터링 로직 구현 가능
    return filters

def main():
    """대시보드 메인 함수"""
    st.title("부안군 감성 분석 대시보드")
    
    # 사이드바 필터 UI
    st.sidebar.header("필터 옵션")
    selected_platform = st.sidebar.selectbox("플랫폼 선택", ["naver_search", "youtube", "twitter"])
    
    # 키워드 입력 UI
    st.sidebar.subheader("키워드 설정")
    region_keyword = st.sidebar.text_input("지역 키워드", "부안")
    additional_keywords = st.sidebar.text_input("추가 키워드 (쉼표로 구분)", "맛집,관광,숙소")
    
    # 페이지 수 설정
    num_pages = st.sidebar.number_input("크롤링할 페이지 수", min_value=1, max_value=10000, value=3)
    
    # JSON 파일 개수 확인
    json_files = glob.glob("data/raw/**/*.json", recursive=True)
    json_counts = Counter([f.split('_')[0].replace("data/raw/", "") for f in json_files])
    for platform, count in json_counts.items():
        st.sidebar.write(f"{platform}: {count}개")
    
    # 감성 분석기 선택
    analyzer_option = st.sidebar.selectbox("감성 분석기 선택", ["Naive Bayes", "KoBERT", "Ensemble"])
    
    # ✅ 데이터 수집 버튼 추가
    if st.sidebar.button("🔄 데이터 수집 실행"):
        if selected_platform == "naver_search":
            # 키워드 리스트 생성
            keywords = [region_keyword]
            if additional_keywords:
                keywords.extend([k.strip() for k in additional_keywords.split(',')])
            
            crawler = NaverSearchAPICrawler(
                keywords=keywords,
                max_pages=num_pages
            )
            crawler.crawl()
            st.success(f"✅ 네이버 크롤링 완료! ({num_pages}페이지)")
        elif selected_platform == "youtube":
            crawler = YouTubeCrawler(keywords=[region_keyword], max_results=30)
            crawler.crawl()
            st.success("✅ 유튜브 크롤링 완료!")
        # 트위터 등 추가 가능
    
    # 데이터 로드
    data_loader = DataLoader()
    data = data_loader.load_data(
        platform=selected_platform, 
        keyword=region_keyword
    )
    
    # 중복 필터링
    unique_data = {item['url']: item for item in data}.values()
    
    # 감성 분석기 초기화
    if analyzer_option == "Naive Bayes":
        sentiment_analyzer = SentimentAnalyzer()
    elif analyzer_option == "KoBERT":
        sentiment_analyzer = KoBERTSentimentAnalyzer()
    elif analyzer_option == "Ensemble":
        sentiment_analyzer = EnsembleSentimentAnalyzer()
    
    # 감성 분석 수행
    for item in unique_data:
        if 'sentiment' not in item or item['sentiment'] is None:
            item['sentiment'], item['confidence'] = analyze_sentiment(item['content'], sentiment_analyzer)

    # 감성 분석 결과 샘플 출력
    st.write("📋 감성 분석 샘플", list(unique_data)[:3])

    # CSV 다운로드 및 분석용 DataFrame 생성
    for item in unique_data:
        if 'platform' not in item:
            item['platform'] = selected_platform  # 사이드바 선택값 활용
        if 'sentiment' not in item:
            item['sentiment'] = 'neutral'  # 기본값으로 'neutral' 설정

    df = pd.DataFrame(unique_data)
    st.download_button("CSV 다운로드", df.to_csv(index=False), file_name="sentiment_results.csv")
    
    # 감성 통계 시각화
    st.subheader("감성 분포")
    sentiments = [item['sentiment'] for item in unique_data]
    sentiment_counts = pd.Series(sentiments).value_counts()
    st.pyplot(sentiment_counts.plot.pie(autopct="%.1f%%", figsize=(5, 5)).figure)
    
    # 긍정 감성 워드클라우드
    st.subheader("긍정 감성 워드클라우드")
    positive_text = " ".join([item["content"] for item in unique_data if item["sentiment"] == "positive"])
    
    if positive_text.strip():  # 긍정 감성 텍스트가 있는지 확인
        wordcloud = WordCloud(font_path="NanumGothic.ttf", background_color='white').generate(positive_text)
        st.image(wordcloud.to_array())
    else:
        st.write("긍정 감성의 텍스트가 없습니다.")
    
    # 시계열 트렌드
    st.subheader("시계열 감성 트렌드")
    plot_time_series(unique_data)
    
    # 지도 시각화
    st.subheader("감성 분포 지도")
    map_ = create_map(unique_data)
    st_folium(map_, width=700, height=500)
    
    # GPT 리포트 생성
    st.subheader("정책 제안 리포트")
    report_generator = GPTReportGenerator(api_key="YOUR_OPENAI_API_KEY")
    report = report_generator.generate_report(df)
    st.text(report)
    
    # PDF 리포트 저장
    pdf_generator = PDFReportGenerator()
    pdf_generator.generate_pdf(report)
    st.success("PDF 리포트가 생성되었습니다.")

if __name__ == "__main__":
    main() 