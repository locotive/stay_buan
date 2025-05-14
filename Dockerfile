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
    firefox-esr \
    xvfb \
    libxi6 \
    libgconf-2-4 \
    && apt-get clean

# Firefox 및 GeckoDriver 설치
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.33.0/geckodriver-v0.33.0-linux64.tar.gz \
    && tar -xvzf geckodriver-v0.33.0-linux64.tar.gz \
    && mv geckodriver /usr/local/bin/ \
    && chmod +x /usr/local/bin/geckodriver \
    && rm geckodriver-v0.33.0-linux64.tar.gz

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

# 환경 변수 설정 - Chromium 사용
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROME_PATH=/usr/lib/chromium/
ENV CHROMIUM_FLAGS="--no-sandbox --headless --disable-gpu --disable-dev-shm-usage"

# Firefox 환경 변수 설정
ENV FIREFOX_BIN=/usr/bin/firefox-esr
ENV GECKODRIVER_PATH=/usr/local/bin/geckodriver

# WebDriver 오류 방지
ENV WDM_LOG_LEVEL=0
ENV WDM_PRINT_FIRST_LINE=false
ENV PYTHONWARNINGS="ignore"

# PyTorch 환경 설정
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV PYTORCH_NO_CUDA_MEMORY_CACHING=1
ENV TRANSFORMERS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Selenium stealth 설치
RUN pip install selenium-stealth

# 앱 실행 명령
CMD ["streamlit", "run", "visualization/dashboard.py"]
