# Stay Buan - 부안 관광 데이터 분석 시스템 코드북

## 프로젝트 개요

이 프로젝트는 부안군 관련 데이터를 다양한 온라인 플랫폼에서 수집하고, 감성 분석을 통해 관광 트렌드를 분석하는 종합적인 데이터 분석 시스템입니다.

## 프로젝트 구조

```
stay_buan/
├── main.py                 # 메인 실행 파일
├── requirements.txt        # Python 패키지 의존성
├── Dockerfile             # Docker 이미지 설정
├── docker-compose.yml     # Docker Compose 설정
├── README.md              # 프로젝트 설명서
├── CODEBOOK.md            # 코드북 (현재 파일)
├── .env                   # 환경 변수 설정
├── .gitignore             # Git 무시 파일 목록
├── crawlers/              # 크롤러 모듈
├── core/                  # 핵심 기능 모듈
├── visualization/         # 시각화 모듈
├── utils/                 # 유틸리티 함수
├── config/                # 설정 파일
├── data/                  # 데이터 저장소
├── logs/                  # 로그 파일
└── reporting/             # 보고서 생성 모듈
```

## 주요 모듈 설명

### 1. main.py
**파일 크기**: 14KB (342줄)
**주요 기능**: 
- 크롤러 실행을 위한 메인 스크립트
- 명령줄 인터페이스 제공
- 병렬 크롤링 지원
- 로깅 시스템 관리

**주요 함수**:
- `setup_logger()`: 로깅 시스템 설정
- `crawl_platform()`: 특정 플랫폼 크롤링 실행
- `save_combined_results()`: 통합 결과 저장
- `main()`: 메인 실행 함수

### 2. crawlers/ 디렉토리
각 플랫폼별 크롤러 모듈들이 포함되어 있습니다.

#### 2.1 naver_api_crawler.py
**파일 크기**: 34KB (731줄)
**기능**: 네이버 검색 API를 통한 데이터 수집
- 뉴스, 블로그, 카페 데이터 수집
- 키워드 기반 검색
- 감성 분석 통합

#### 2.2 youtube.py
**파일 크기**: 14KB (311줄)
**기능**: YouTube API를 통한 비디오 및 댓글 수집
- 비디오 메타데이터 수집
- 댓글 데이터 수집
- 감성 분석 적용

#### 2.3 google_search.py
**파일 크기**: 21KB (457줄)
**기능**: Google Custom Search API를 통한 웹 검색
- 웹 검색 결과 수집
- API 사용량 제한 관리
- 키워드 기반 필터링

#### 2.4 dcinside.py
**파일 크기**: 20KB (468줄)
**기능**: DCinside 게시글 및 댓글 수집
- 게시글 내용 수집
- 댓글 데이터 수집
- robots.txt 정책 준수 옵션

#### 2.5 buan_gov.py
**파일 크기**: 28KB (615줄)
**기능**: 부안군청 홈페이지 데이터 수집
- 공지사항, 보도자료 수집
- Selenium 기반 동적 크롤링
- 첨부파일 URL 수집

### 3. core/ 디렉토리
핵심 기능 모듈들이 포함되어 있습니다.

#### 3.1 base_crawler.py
**파일 크기**: 5.2KB (146줄)
**기능**: 크롤러 기본 클래스
- 공통 크롤링 기능 정의
- 데이터 저장 표준화
- 에러 처리 기본 구현

#### 3.2 감성 분석 모듈들
- `sentiment_analysis.py`: 감성 분석 기본 인터페이스
- `sentiment_analysis_kobert.py`: KoBERT 모델 기반 분석
- `sentiment_analysis_ensemble.py`: 앙상블 모델 기반 분석
- `sentiment_analysis_kcbert_large.py`: KC-BERT-Large 모델
- `sentiment_analysis_kcelectra.py`: KC-Electra 모델
- `sentiment_analysis_kosentencebert.py`: Ko-SentenceBERT 모델
- `model_cache.py`: 모델 캐싱 시스템

### 4. visualization/ 디렉토리

#### 4.1 dashboard.py
**파일 크기**: 85KB (1782줄)
**기능**: Streamlit 기반 대시보드
- 실시간 데이터 시각화
- 크롤링 제어 인터페이스
- 감성 분석 결과 표시
- 보고서 생성 기능

### 5. utils/ 디렉토리
유틸리티 함수들을 포함합니다.

### 6. config/ 디렉토리
설정 파일들을 포함합니다.

### 7. reporting/ 디렉토리
보고서 생성 모듈들을 포함합니다.

## 데이터 구조

### 수집 데이터 형식
각 크롤러는 다음과 같은 공통 데이터 구조를 반환합니다:

```python
{
    "platform": "플랫폼명",
    "title": "제목",
    "content": "내용",
    "url": "원본 URL",
    "timestamp": "수집 시간",
    "sentiment": {
        "label": "긍정/중립/부정",
        "score": "신뢰도 점수"
    },
    "metadata": {
        # 플랫폼별 추가 정보
    }
}
```

### 저장 위치
- `data/raw/`: 원본 수집 데이터
- `data/processed/`: 전처리된 데이터
- `logs/`: 로그 파일들

## 주요 기능

### 1. 다중 플랫폼 크롤링
- 네이버, YouTube, Google, DCinside, 부안군청
- 병렬 처리 지원
- 실시간 진행 상황 모니터링

### 2. 감성 분석
- 다중 모델 앙상블 시스템
- 한국어 특화 모델 사용
- 실시간 분석 및 캐싱

### 3. 시각화 대시보드
- Streamlit 기반 웹 인터페이스
- 실시간 데이터 업데이트
- 인터랙티브 차트 및 그래프

### 4. 자동화된 보고서 생성
- PDF 형식 보고서 생성
- 시각화 결과 포함
- 정기적 자동 생성

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

## API 키 설정

프로젝트 실행을 위해 다음 API 키들이 필요합니다:

```env
# 네이버 API
NAVER_CLIENT_ID=your_client_id
NAVER_CLIENT_SECRET=your_client_secret

# 유튜브 API
YOUTUBE_API_KEY=your_youtube_api_key

# 구글 커스텀 검색 API
GOOGLE_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id
```

## 실행 방법

### 1. Python 환경에서 실행
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# 크롤러 실행
python main.py --platform all --keywords "부안 관광"

# 대시보드 실행
streamlit run visualization/dashboard.py
```

### 2. Docker 환경에서 실행
```bash
# 이미지 빌드
docker build -t stay-buan .

# 컨테이너 실행
docker run -p 8501:8501 --env-file .env stay-buan
```

## 성능 최적화

### 1. 병렬 처리
- 멀티스레딩을 통한 동시 크롤링
- CPU 코어 수에 따른 자동 조정

### 2. 모델 캐싱
- 감성 분석 모델 로딩 최적화
- 결과 캐싱을 통한 중복 계산 방지

### 3. 메모리 관리
- 대용량 데이터 처리 시 청크 단위 처리
- 가비지 컬렉션 최적화

## 에러 처리

### 1. 네트워크 에러
- 재시도 메커니즘 구현
- 타임아웃 설정
- 연결 실패 시 로깅

### 2. API 제한
- 요청 빈도 제한
- 사용량 모니터링
- 자동 재시도 스케줄링

### 3. 데이터 검증
- 수집 데이터 품질 검사
- 필수 필드 누락 확인
- 형식 검증

## 확장성

### 1. 새로운 플랫폼 추가
- `base_crawler.py` 상속
- 표준 인터페이스 구현
- 설정 파일 추가

### 2. 새로운 분석 모델 추가
- 감성 분석 인터페이스 구현
- 모델 등록 시스템
- 성능 비교 자동화

### 3. 새로운 시각화 추가
- Streamlit 컴포넌트 확장
- 차트 타입 추가
- 대시보드 레이아웃 커스터마이징

## 보안 고려사항

### 1. API 키 보안
- 환경 변수 사용
- .env 파일 gitignore
- Docker secrets 활용

### 2. 데이터 보안
- 개인정보 필터링
- 민감 데이터 암호화
- 접근 권한 관리

### 3. 웹 크롤링 정책
- robots.txt 준수
- 요청 빈도 제한
- 웹사이트 정책 확인

## 모니터링 및 로깅

### 1. 로그 시스템
- 파일 기반 로깅
- 레벨별 로그 분리
- 로그 로테이션

### 2. 성능 모니터링
- 실행 시간 측정
- 메모리 사용량 추적
- API 사용량 모니터링

### 3. 에러 추적
- 상세한 에러 로깅
- 스택 트레이스 저장
- 알림 시스템

## 테스트

### 1. 단위 테스트
- 각 모듈별 테스트 케이스
- 모킹을 통한 외부 의존성 격리
- 자동화된 테스트 실행

### 2. 통합 테스트
- 전체 워크플로우 테스트
- 실제 API 호출 테스트
- 성능 테스트

### 3. 사용자 테스트
- 대시보드 사용성 테스트
- 크롤링 결과 검증
- 보고서 품질 확인

## 배포

### 1. Docker 배포
- 멀티스테이지 빌드
- 최적화된 이미지 크기
- 환경별 설정 분리

### 2. 클라우드 배포
- AWS/GCP/Azure 지원
- 자동 스케일링
- 로드 밸런싱

### 3. CI/CD
- GitHub Actions 자동화
- 자동 테스트 및 배포
- 롤백 메커니즘

## 유지보수

### 1. 코드 품질
- PEP 8 스타일 가이드 준수
- 타입 힌트 사용
- 문서화 주석

### 2. 버전 관리
- Git 기반 버전 관리
- 브랜치 전략
- 릴리즈 노트

### 3. 의존성 관리
- 정기적 패키지 업데이트
- 보안 취약점 스캔
- 호환성 테스트

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여 가이드

### 1. 코드 기여
- Fork & Pull Request 방식
- 코드 리뷰 필수
- 테스트 코드 작성

### 2. 문서 기여
- README 업데이트
- 코드북 보완
- 사용자 가이드 작성

### 3. 이슈 리포트
- 버그 리포트
- 기능 요청
- 개선 제안

---

이 코드북은 프로젝트의 기술적 구조와 구현 세부사항을 설명합니다. 추가 질문이나 개선 제안이 있으시면 언제든지 문의해 주세요. 