# Naver Search API Crawler

이 프로젝트는 네이버 검색 API를 사용하여 뉴스, 블로그, 카페 등의 데이터를 수집하고, 수집된 데이터를 JSON 형식으로 저장하는 크롤러입니다. 또한, 수집된 데이터를 시각화하는 대시보드를 제공합니다.

## 프로젝트 설정

### 1. 환경 설정

- Python 3.9 이상이 필요합니다.
- 필요한 Python 패키지는 `requirements.txt`에 명시되어 있습니다.

### 2. 환경 변수 설정

- `.env` 파일을 프로젝트 루트 디렉토리에 생성하고, 네이버 API 클라이언트 ID와 시크릿을 설정합니다.
plaintext
NAVER_CLIENT_ID=your_client_id
NAVER_CLIENT_SECRET=your_client_secret


## 설치 및 실행

### 1. Python 환경에서 실행

1. 가상 환경 생성 및 활성화:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows의 경우: venv\Scripts\activate
   ```

2. 필요한 패키지 설치:

   ```bash
   pip install -r requirements.txt
   ```

3. 크롤러 실행:

   ```bash
   python crawlers/naver_api_crawler.py
   ```

4. 대시보드 실행:

   ```bash
   streamlit run visualization/dashboard.py
   ```

### 2. Docker를 사용하여 실행

1. Docker 이미지 빌드:

   ```bash
   docker build -t naver-crawler .
   ```

2. Docker 컨테이너 실행:

   ```bash
   docker run -p 8501:8501 --env-file .env naver-crawler
   ```

   - `--env-file .env` 옵션을 사용하여 환경 변수를 Docker 컨테이너에 전달합니다.

3. 웹 브라우저에서 대시보드 확인:

   - [http://localhost:8501](http://localhost:8501)로 접속하여 대시보드를 확인할 수 있습니다.

## 디렉토리 구조

- `crawlers/`: 크롤러 코드가 포함된 디렉토리
- `visualization/`: 대시보드 코드가 포함된 디렉토리
- `data/`: 수집된 데이터가 저장되는 디렉토리
- `.env`: 환경 변수 설정 파일
- `requirements.txt`: 필요한 Python 패키지 목록
- `Dockerfile`: Docker 이미지 빌드 설정 파일

## 주의사항

- 네이버 API의 사용량 제한에 주의하세요. API 호출 횟수가 초과되면 데이터 수집이 중단될 수 있습니다.
- 크롤러 실행 시, 수집된 데이터는 `data/raw/` 디렉토리에 JSON 형식으로 저장됩니다.

이 문서를 참고하여 프로젝트를 설정하고 실행할 수 있습니다. 추가적인 도움이 필요하시면 언제든지 문의해 주세요!