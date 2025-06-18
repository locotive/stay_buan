# Stay Buan - 부안 관광 데이터 분석 시스템

## 프로젝트 개요

이 프로젝트는 다양한 온라인 플랫폼(네이버, 유튜브, 구글, DCinside, 부안군청 등)에서 부안군 관련 데이터를 수집하고, 수집된 데이터의 감성을 분석하여 시각화하는 종합적인 데이터 분석 시스템입니다. 

### 주요 성과
- **실제 정책 활용**: 교수님의 세미나 발표를 통해 검증받은 시스템
- **지자체 관심**: 다른 지자체 관계자들의 높은 관심도
- **본선 진출**: 지자체 관계자들과 함께하는 본선 발표 준비 중
- **데이터 기반 의사결정**: 부안군 관광 정책 수립에 직접 활용

### 핵심 기능
- **다중 플랫폼 데이터 수집**: 5개 주요 플랫폼에서 실시간 데이터 수집
- **감성 분석**: 한국어 특화 다중 모델 앙상블 시스템
- **실시간 대시보드**: Streamlit 기반 인터랙티브 시각화
- **자동화된 보고서**: PDF 형식의 정기 보고서 생성
- **Docker 기반 운영**: 안정적인 배포 및 운영 환경

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

### 3. Windows 사용자를 위한 간편 실행

Windows 사용자는 `start_dashboard.bat` 파일을 더블 클릭하여 Docker 기반으로 시스템을 실행할 수 있습니다.

## 지원 플랫폼

이 프로젝트는 다음과 같은 플랫폼에서 데이터를 수집할 수 있습니다:

1. **네이버 (naver)**: 네이버 검색 API를 사용하여 뉴스, 블로그, 카페 등의 데이터를 수집합니다.
2. **유튜브 (youtube)**: 유튜브 API를 사용하여 비디오 정보와 댓글을 수집합니다.
3. **구글 (google)**: 구글 커스텀 검색 API를 사용하여 웹 검색 결과를 수집합니다.
4. **DCinside (dcinside)**: DCinside의 게시글과 댓글을 수집합니다. (robots.txt로 인해 제한됨, `--ignore-robots` 옵션 필요)
5. **부안군청 (buan)**: 부안군청 홈페이지의 공지사항, 보도자료, 군정소식, 고시공고 게시판을 수집합니다.

## 특별 기능

### 1. 감성 분석

수집된 모든 텍스트 데이터(제목, 내용, 댓글 등)는 한국어 감성 분석 모델을 통해 분석되어 긍정(positive), 중립(neutral), 부정(negative) 범주로 분류됩니다.

**사용된 모델들:**
- KoBERT
- KC-BERT-Large
- KC-Electra
- Ko-SentenceBERT
- 앙상블 모델 (다중 모델 결합)

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
- 웹사이트 정책을 위반할 경우 법적 문제가 발생할 수 있으므로 주의하세요.

### 5. 병렬 크롤링

여러 플랫폼을 동시에 크롤링하여 시간을 절약할 수 있습니다:

- 병렬 처리 옵션을 통해 모든 플랫폼을 동시에 크롤링합니다.
- 실시간 진행 상황을 시각적으로 확인할 수 있습니다.
- 통합된 결과를 하나의 파일로 저장합니다.

## 실제 활용 사례

### 1. 정책 수립 지원
- 부안군 관광 정책 수립에 직접 활용
- 데이터 기반 의사결정 지원
- 관광 트렌드 분석 및 예측

### 2. 세미나 발표
- 교수님의 세미나 발표를 통한 검증
- 다른 지자체 관계자들의 높은 관심도
- 본선 발표 진출

### 3. 확장 가능성
- 다른 지자체 적용 가능
- 다양한 정책 영역으로 확장
- 협업 및 공유 방안 모색

## 디렉토리 구조

```
stay_buan/
├── main.py                 # 메인 실행 파일
├── requirements.txt        # Python 패키지 의존성
├── Dockerfile             # Docker 이미지 설정
├── docker-compose.yml     # Docker Compose 설정
├── README.md              # 프로젝트 설명서 (현재 파일)
├── CODEBOOK.md            # 코드북
├── .env                   # 환경 변수 설정
├── .gitignore             # Git 무시 파일 목록
├── crawlers/              # 크롤러 모듈
│   ├── naver_api_crawler.py
│   ├── youtube.py
│   ├── google_search.py
│   ├── dcinside.py
│   └── buan_gov.py
├── core/                  # 핵심 기능 모듈
│   ├── base_crawler.py
│   ├── sentiment_analysis*.py
│   └── model_cache.py
├── visualization/         # 시각화 모듈
│   └── dashboard.py
├── utils/                 # 유틸리티 함수
├── config/                # 설정 파일
├── data/                  # 데이터 저장소
├── logs/                  # 로그 파일
└── reporting/             # 보고서 생성 모듈
```

## 기술 스택

### Backend
- **Python 3.9+**: 메인 개발 언어
- **Pandas**: 데이터 처리
- **NumPy**: 수치 계산

### 크롤링
- **Selenium**: 동적 웹페이지 크롤링
- **Requests**: HTTP 요청
- **BeautifulSoup**: HTML 파싱

### 감성 분석
- **Transformers**: BERT 모델
- **Torch**: 딥러닝 프레임워크
- **KoNLPy**: 한국어 자연어 처리

### 시각화
- **Streamlit**: 웹 대시보드
- **Plotly**: 인터랙티브 차트
- **Matplotlib**: 기본 차트

### 인프라
- **Docker**: 컨테이너화
- **Docker Compose**: 멀티 컨테이너 관리

## 주의사항

- API 사용량 제한에 주의하세요. API 호출 횟수가 초과되면 데이터 수집이 중단될 수 있습니다.
- 크롤러 실행 시, 수집된 데이터는 `data/raw/` 디렉토리에 JSON 형식으로 저장됩니다.
- 일부 웹사이트의 로봇 정책이 변경될 경우 크롤러가 정상 작동하지 않을 수 있습니다.
- Selenium을 사용하는 부안군청 크롤러는 ChromeDriver가 필요합니다. 자동 설치가 실패할 경우 수동으로 설치해야 할 수 있습니다.

## 향후 계획

### 1. 시스템 개선
- 추가 감성 분석 모델 도입
- 실시간 알림 시스템 구축
- 모바일 대시보드 개발

### 2. 확장 방안
- 다른 지자체 적용
- 다양한 정책 영역 확장
- API 서비스 제공

### 3. 협업 방안
- 지자체 간 데이터 공유
- 표준화된 분석 프레임워크 구축
- 오픈소스 프로젝트로 발전

## 문의 및 지원

이 문서를 참고하여 프로젝트를 설정하고 실행할 수 있습니다. 추가적인 도움이 필요하시면 언제든지 문의해 주세요!

---

**프로젝트명**: Stay Buan - 부안 관광 데이터 분석 시스템  
**개발팀**: Stay_Buan  
**버전**: 1.0.0  
**최종 업데이트**: 2024년 12월