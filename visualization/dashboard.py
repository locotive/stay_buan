import streamlit as st
import folium
from streamlit_folium import st_folium
from utils.data_loader import DataLoader
from core.sentiment_analysis_kobert import KoBERTSentimentAnalyzer
from reporting.report_generator_gpt import GPTReportGenerator
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from folium.plugins import MarkerCluster

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
    """시계열 트렌드 시각화"""
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # 감성별로 그룹화하여 일별 카운트
    sentiment_trend = df.resample('D').apply(lambda x: x['sentiment'].value_counts()).unstack().fillna(0)
    
    # 시계열 그래프 그리기
    plt.figure(figsize=(10, 5))
    sentiment_trend.plot(ax=plt.gca())
    plt.title("일별 감성 트렌드")
    plt.xlabel("날짜")
    plt.ylabel("감성 수")
    plt.legend(title="감성")
    st.pyplot(plt)

def main():
    """대시보드 메인 함수"""
    st.title("부안군 감성 분석 대시보드")
    
    # 사이드바 필터 UI
    st.sidebar.header("필터 옵션")
    selected_platform = st.sidebar.selectbox("플랫폼 선택", ["naver_blog", "youtube", "twitter"])
    selected_keyword = st.sidebar.text_input("검색 키워드", "부안")
    start_date = st.sidebar.date_input("시작 날짜", datetime(2023, 1, 1))
    end_date = st.sidebar.date_input("종료 날짜", datetime.now())
    
    # 사용자 질문 입력
    user_query = st.text_input("질문 입력", "최근 부안군 관광에 대한 부정 감성 많았나요?")
    
    # 데이터 로드
    data_loader = DataLoader()
    data = data_loader.load_data(
        platform=selected_platform, 
        keyword=selected_keyword, 
        start_date=start_date, 
        end_date=end_date
    )
    
    # 중복 필터링
    unique_data = {item['url']: item for item in data}.values()
    
    # 감성 분석
    sentiment_analyzer = KoBERTSentimentAnalyzer()
    for item in unique_data:
        if 'sentiment' not in item or item['sentiment'] is None:
            item['sentiment'], item['confidence'] = analyze_sentiment(item['content'], sentiment_analyzer)
    
    # 사용자 질문에 따른 필터링
    if "부정" in user_query:
        filtered_data = [item for item in unique_data if item['sentiment'] == 'negative']
    else:
        filtered_data = unique_data
    
    # CSV 다운로드
    df = pd.DataFrame(filtered_data)
    st.download_button("CSV 다운로드", df.to_csv(index=False), file_name="sentiment_results.csv")
    
    # 감성 통계 시각화
    st.subheader("감성 분포")
    sentiments = [item['sentiment'] for item in filtered_data]
    sentiment_counts = pd.Series(sentiments).value_counts()
    st.pyplot(sentiment_counts.plot.pie(autopct="%.1f%%", figsize=(5, 5)).figure)
    
    # 워드클라우드
    st.subheader("긍정 감성 워드클라우드")
    positive_text = " ".join([item["content"] for item in filtered_data if item["sentiment"] == "positive"])
    wordcloud = WordCloud(font_path="NanumGothic.ttf", background_color='white').generate(positive_text)
    st.image(wordcloud.to_array())
    
    # 시계열 트렌드
    st.subheader("시계열 감성 트렌드")
    plot_time_series(filtered_data)
    
    # 지도 시각화
    st.subheader("감성 분포 지도")
    map_ = create_map(filtered_data)
    st_folium(map_, width=700, height=500)
    
    # GPT 리포트 생성
    st.subheader("정책 제안 리포트")
    report_generator = GPTReportGenerator(api_key="YOUR_OPENAI_API_KEY")
    report = report_generator.generate_report(df)
    st.text(report)

if __name__ == "__main__":
    main() 