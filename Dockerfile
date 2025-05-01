FROM python:3.9-slim

# 🛠 Java + 빌드 도구 설치 (Okt 실행에 필요)
RUN apt-get update && apt-get install -y \
    default-jdk \
    curl \
    git \
    build-essential \
    wget \
    unzip \
    gnupg \
    chromium \
    chromium-driver \
    && apt-get clean

# JAVA_HOME 설정 (default-jdk가 설치한 경로로 자동 설정됨)
ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
ENV PATH="$JAVA_HOME/bin:$PATH"

# 앱 작업 디렉토리 설정
WORKDIR /app

# 📦 파이썬 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사 및 PYTHONPATH 설정
COPY . .
ENV PYTHONPATH=/app

# 포트 노출
EXPOSE 8501

# Chrome 설치
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver

# 환경 변수 설정
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROME_PATH=/usr/lib/chromium/
ENV CHROMIUM_FLAGS="--no-sandbox --headless --disable-gpu --disable-dev-shm-usage"

# ChromeDriver 설치
RUN CHROMEDRIVER_VERSION=`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE` \
    && wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip \
    && unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/ \
    && rm /tmp/chromedriver.zip

RUN pip install selenium-stealth

# 앱 실행 명령
CMD ["streamlit", "run", "visualization/dashboard.py"]
