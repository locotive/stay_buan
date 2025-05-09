# 부안군 감성 분석 대시보드

이 프로젝트는 다양한 온라인 플랫폼(네이버, 유튜브, 구글, DCinside, FMKorea, 부안군청 등)에서 부안군 관련 데이터를 수집하고, 수집된 데이터의 감성을 분석하여 시각화하는 대시보드를 제공합니다.

## 프로젝트 설정

### 1. 환경 설정

- Python 3.9 이상이 필요합니다.
- 필요한 Python 패키지는 `requirements.txt`에 명시되어 있습니다.
- Selenium을 사용하기 위해 Chrome 브라우저와 해당 버전의 ChromeDriver가 필요합니다.

### 2. 환경 변수 설정

- `.env` 파일을 프로젝트 루트 디렉토리에 생성하고, 필요한 API 키를 설정합니다.
```plaintext
# 네이버 API
NAVER_CLIENT_ID=your_client_id
NAVER_CLIENT_SECRET=your_client_secret

# 유튜브 API
YOUTUBE_API_KEY=your_youtube_api_key

# 구글 커스텀 검색 API
GOOGLE_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id
```

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

3. 크롤러 실행 (명령줄 인터페이스):

   ```bash
   # 특정 플랫폼 데이터 수집
   python main.py --platform naver --keywords 부안 축제 여행
   python main.py --platform youtube --keywords 부안 맛집 --pages 20 --comments 30
   python main.py --platform google --keywords 부안 관광 --pages 5 --google-query-limit 100
   python main.py --platform dcinside --keywords 부안 변산반도 --pages 5 --comments 50 --ignore-robots
   python main.py --platform fmkorea --keywords 부안 여행후기 --pages 3 --ignore-robots
   python main.py --platform buan --keywords 부안 행사 --pages 3
   
   # 모든 플랫폼 데이터 수집
   python main.py --platform all --keywords 부안 바다 --pages 2 --ignore-robots
   
   # 감성 분석 없이 데이터만 수집
   python main.py --platform naver --keywords 부안 --no-sentiment
   ```

4. 통합 대시보드 실행:

   ```bash
   streamlit run visualization/dashboard.py
   ```
   
   대시보드는 크롤링 기능과 데이터 분석 기능이 탭으로 구분되어 있습니다:
   - **크롤링 탭**: 키워드 설정, 플랫폼 선택, 크롤링 진행 상황 표시, 결과 통계 확인
   - **데이터 분석 탭**: 데이터셋 선택, 감성 분석 수행, 시각화 및 리포트 생성

### 2. Docker를 사용하여 실행

1. Docker 이미지 빌드:

   ```bash
   docker build -t buan-sentiment-dashboard .
   ```

2. Docker 컨테이너 실행:

   ```bash
   docker run -p 8501:8501 --env-file .env buan-sentiment-dashboard
   ```

   - `--env-file .env` 옵션을 사용하여 환경 변수를 Docker 컨테이너에 전달합니다.

3. 웹 브라우저에서 대시보드 확인:

   - [http://localhost:8501](http://localhost:8501)로 접속하여 대시보드를 확인할 수 있습니다.

## 지원 플랫폼

이 프로젝트는 다음과 같은 플랫폼에서 데이터를 수집할 수 있습니다:

1. **네이버 (naver)**: 네이버 검색 API를 사용하여 뉴스, 블로그, 카페 등의 데이터를 수집합니다.
2. **유튜브 (youtube)**: 유튜브 API를 사용하여 비디오 정보와 댓글을 수집합니다.
3. **구글 (google)**: 구글 커스텀 검색 API를 사용하여 웹 검색 결과를 수집합니다.
4. **DCinside (dcinside)**: DCinside의 게시글과 댓글을 수집합니다. (robots.txt로 인해 제한됨, `--ignore-robots` 옵션 필요)
5. **FMKorea (fmkorea)**: FMKorea의 게시글과 댓글을 수집합니다. (검색 기능은 robots.txt로 제한됨, `--ignore-robots` 옵션 사용 가능)
6. **부안군청 (buan)**: 부안군청 홈페이지의 공지사항, 보도자료, 군정소식, 고시공고 게시판을 수집합니다.

## 특별 기능

### 1. 감성 분석

수집된 모든 텍스트 데이터(제목, 내용, 댓글 등)는 한국어 감성 분석 모델을 통해 분석되어 긍정(positive), 중립(neutral), 부정(negative) 범주로 분류됩니다.

### 2. 구글 API 사용량 제한 및 알림

구글 검색 API는 무료 계정에서 일일 100회의 쿼리 제한이 있습니다. 이 제한을 관리하기 위한 기능이 구현되어 있습니다:

- `--google-query-limit` 옵션으로 일일 최대 쿼리 수를 설정할 수 있습니다.
- 사용량이 80% 이상이 되면 경고 메시지가 표시됩니다.
- 한도에 도달하면 추가 쿼리가 중단됩니다.
- 키워드 변형과 페이지 수가 API 한도에 맞게 자동으로 조정됩니다.

### 3. 부안군청 홈페이지 크롤링

부안군청 홈페이지는 Selenium을 사용하여 동적 콘텐츠를 크롤링합니다:

- 공지사항, 보도자료, 군정소식, 고시공고 게시판에서 데이터를 수집합니다.
- 첨부파일 URL도 함께 수집됩니다.
- 키워드 검색을 통해 관련 게시글만 필터링합니다.

### 4. robots.txt 정책 준수

각 웹사이트의 robots.txt 정책을 확인하고 준수할 수 있는 기능이 구현되어 있습니다:

- DCinside는 robots.txt에서 일반 크롤러의 모든 접근을 차단하고 있습니다. 이 크롤러를 사용하려면 `--ignore-robots` 옵션이 필요합니다.
- FMKorea는 검색 기능(`search_keyword=` 파라미터)에 대한 접근을 차단하고 있습니다. 검색을 사용하려면 `--ignore-robots` 옵션이 필요합니다.
- 웹사이트 정책을 위반할 경우 법적 문제가 발생할 수 있으므로 주의하세요.

### 5. 병렬 크롤링

여러 플랫폼을 동시에 크롤링하여 시간을 절약할 수 있습니다:

- 병렬 처리 옵션을 통해 모든 플랫폼을 동시에 크롤링합니다.
- 실시간 진행 상황을 시각적으로 확인할 수 있습니다.
- 통합된 결과를 하나의 파일로 저장합니다.

## 디렉토리 구조

- `core/`: 기본 크롤러 및 감성 분석 코드
- `crawlers/`: 각 플랫폼별 크롤러 코드
- `visualization/`: 데이터 시각화 및 대시보드 코드
- `data/`: 수집된 데이터가 저장되는 디렉토리
- `utils/`: 유틸리티 함수 모음
- `.env`: 환경 변수 설정 파일
- `requirements.txt`: 필요한 Python 패키지 목록
- `Dockerfile`: Docker 이미지 빌드 설정 파일
- `main.py`: 크롤러 실행을 위한 메인 스크립트

## 주의사항

- API 사용량 제한에 주의하세요. API 호출 횟수가 초과되면 데이터 수집이 중단될 수 있습니다.
- 크롤러 실행 시, 수집된 데이터는 `data/raw/` 디렉토리에 JSON 형식으로 저장됩니다.
- 일부 웹사이트의 로봇 정책이 변경될 경우 크롤러가 정상 작동하지 않을 수 있습니다.
- Selenium을 사용하는 부안군청 크롤러는 ChromeDriver가 필요합니다. 자동 설치가 실패할 경우 수동으로 설치해야 할 수 있습니다.
- DCinside, FMKorea와 같은 웹사이트는 robots.txt 정책에 따라 크롤링이 제한될 수 있습니다. 필요한 경우 `--ignore-robots` 옵션을 사용하되, 웹사이트 이용 약관과 정책을 위반하지 않도록 주의하세요.

이 문서를 참고하여 프로젝트를 설정하고 실행할 수 있습니다. 추가적인 도움이 필요하시면 언제든지 문의해 주세요!