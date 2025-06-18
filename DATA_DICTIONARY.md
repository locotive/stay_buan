# Stay Buan - 데이터 정의서 (Data Dictionary)

## 개요

이 문서는 Stay Buan 프로젝트에서 수집하는 모든 데이터의 구조와 각 필드의 정의를 설명합니다. 각 플랫폼별로 수집되는 데이터의 형식과 의미를 상세히 기술합니다.

## 공통 데이터 구조

모든 플랫폼에서 수집되는 데이터는 다음과 같은 기본 구조를 따릅니다:

### 기본 필드

| 필드명 | 데이터 타입 | 필수 여부 | 설명 |
|--------|-------------|-----------|------|
| `platform` | String | 필수 | 데이터 수집 플랫폼명 (naver, youtube, google, dcinside, buan) |
| `title` | String | 필수 | 게시글/콘텐츠 제목 |
| `content` | String | 필수 | 게시글/콘텐츠 본문 내용 |
| `url` | String | 필수 | 원본 콘텐츠 URL |
| `timestamp` | String | 필수 | 데이터 수집 시간 (ISO 8601 형식) |
| `sentiment` | Object | 선택 | 감성 분석 결과 |
| `metadata` | Object | 선택 | 플랫폼별 추가 정보 |

### 감성 분석 결과 구조

```json
{
  "sentiment": {
    "label": "긍정|중립|부정",
    "score": 0.0-1.0,
    "model": "사용된 모델명"
  }
}
```

| 필드명 | 데이터 타입 | 설명 |
|--------|-------------|------|
| `label` | String | 감성 분석 결과 레이블 (긍정/중립/부정) |
| `score` | Float | 신뢰도 점수 (0.0-1.0, 높을수록 신뢰도 높음) |
| `model` | String | 사용된 감성 분석 모델명 |

## 플랫폼별 데이터 구조

### 1. 네이버 (naver)

네이버 검색 API를 통해 수집되는 뉴스, 블로그, 카페 데이터입니다.

#### 기본 구조
```json
{
  "platform": "naver",
  "title": "제목",
  "content": "내용",
  "url": "원본 URL",
  "timestamp": "2024-12-XX",
  "sentiment": {...},
  "metadata": {
    "source": "뉴스|블로그|카페",
    "author": "작성자",
    "publish_date": "발행일",
    "category": "카테고리",
    "description": "요약",
    "thumbnail": "썸네일 URL"
  }
}
```

#### 메타데이터 필드

| 필드명 | 데이터 타입 | 설명 |
|--------|-------------|------|
| `source` | String | 콘텐츠 출처 (뉴스/블로그/카페) |
| `author` | String | 콘텐츠 작성자 |
| `publish_date` | String | 원본 발행일 |
| `category` | String | 콘텐츠 카테고리 |
| `description` | String | 콘텐츠 요약 |
| `thumbnail` | String | 썸네일 이미지 URL |

### 2. 유튜브 (youtube)

YouTube API를 통해 수집되는 비디오 및 댓글 데이터입니다.

#### 비디오 데이터 구조
```json
{
  "platform": "youtube",
  "title": "비디오 제목",
  "content": "비디오 설명",
  "url": "비디오 URL",
  "timestamp": "2024-12-XX",
  "sentiment": {...},
  "metadata": {
    "video_id": "비디오 ID",
    "channel_name": "채널명",
    "channel_id": "채널 ID",
    "publish_date": "업로드 날짜",
    "view_count": 12345,
    "like_count": 123,
    "comment_count": 45,
    "duration": "PT10M30S",
    "tags": ["태그1", "태그2"],
    "category": "카테고리",
    "thumbnail": "썸네일 URL",
    "comments": [...]
  }
}
```

#### 댓글 데이터 구조
```json
{
  "platform": "youtube",
  "title": "댓글 내용",
  "content": "댓글 내용",
  "url": "댓글 URL",
  "timestamp": "2024-12-XX",
  "sentiment": {...},
  "metadata": {
    "comment_id": "댓글 ID",
    "video_id": "비디오 ID",
    "author": "댓글 작성자",
    "publish_date": "댓글 작성일",
    "like_count": 5,
    "reply_count": 2,
    "is_reply": false
  }
}
```

#### 메타데이터 필드

| 필드명 | 데이터 타입 | 설명 |
|--------|-------------|------|
| `video_id` | String | YouTube 비디오 고유 ID |
| `channel_name` | String | 채널명 |
| `channel_id` | String | 채널 고유 ID |
| `publish_date` | String | 비디오 업로드 날짜 |
| `view_count` | Integer | 조회수 |
| `like_count` | Integer | 좋아요 수 |
| `comment_count` | Integer | 댓글 수 |
| `duration` | String | 비디오 길이 (ISO 8601 형식) |
| `tags` | Array | 비디오 태그 목록 |
| `category` | String | 비디오 카테고리 |
| `thumbnail` | String | 썸네일 이미지 URL |
| `comments` | Array | 수집된 댓글 목록 |

### 3. 구글 (google)

Google Custom Search API를 통해 수집되는 웹 검색 결과 데이터입니다.

#### 기본 구조
```json
{
  "platform": "google",
  "title": "페이지 제목",
  "content": "페이지 요약",
  "url": "페이지 URL",
  "timestamp": "2024-12-XX",
  "sentiment": {...},
  "metadata": {
    "display_link": "표시 URL",
    "snippet": "검색 결과 요약",
    "pagemap": {...},
    "search_query": "검색 쿼리",
    "position": 1
  }
}
```

#### 메타데이터 필드

| 필드명 | 데이터 타입 | 설명 |
|--------|-------------|------|
| `display_link` | String | 검색 결과에 표시되는 URL |
| `snippet` | String | 검색 결과 요약 |
| `pagemap` | Object | 페이지 메타데이터 |
| `search_query` | String | 검색에 사용된 쿼리 |
| `position` | Integer | 검색 결과 순위 |

### 4. DCinside (dcinside)

DCinside 게시글 및 댓글 데이터입니다.

#### 게시글 데이터 구조
```json
{
  "platform": "dcinside",
  "title": "게시글 제목",
  "content": "게시글 내용",
  "url": "게시글 URL",
  "timestamp": "2024-12-XX",
  "sentiment": {...},
  "metadata": {
    "gallery": "갤러리명",
    "post_id": "게시글 ID",
    "author": "작성자",
    "publish_date": "작성일",
    "view_count": 123,
    "recommend_count": 5,
    "comment_count": 10,
    "comments": [...]
  }
}
```

#### 댓글 데이터 구조
```json
{
  "platform": "dcinside",
  "title": "댓글 내용",
  "content": "댓글 내용",
  "url": "댓글 URL",
  "timestamp": "2024-12-XX",
  "sentiment": {...},
  "metadata": {
    "gallery": "갤러리명",
    "post_id": "게시글 ID",
    "comment_id": "댓글 ID",
    "author": "댓글 작성자",
    "publish_date": "댓글 작성일",
    "recommend_count": 2,
    "is_reply": false
  }
}
```

#### 메타데이터 필드

| 필드명 | 데이터 타입 | 설명 |
|--------|-------------|------|
| `gallery` | String | DCinside 갤러리명 |
| `post_id` | String | 게시글 고유 ID |
| `author` | String | 작성자명 |
| `publish_date` | String | 작성일 |
| `view_count` | Integer | 조회수 |
| `recommend_count` | Integer | 추천수 |
| `comment_count` | Integer | 댓글 수 |
| `comments` | Array | 수집된 댓글 목록 |

### 5. 부안군청 (buan)

부안군청 홈페이지의 공지사항, 보도자료 등입니다.

#### 기본 구조
```json
{
  "platform": "buan",
  "title": "공지사항 제목",
  "content": "공지사항 내용",
  "url": "공지사항 URL",
  "timestamp": "2024-12-XX",
  "sentiment": {...},
  "metadata": {
    "board_type": "공지사항|보도자료|군정소식|고시공고",
    "post_id": "게시글 ID",
    "author": "작성자",
    "publish_date": "작성일",
    "view_count": 123,
    "attachments": [...],
    "category": "카테고리"
  }
}
```

#### 메타데이터 필드

| 필드명 | 데이터 타입 | 설명 |
|--------|-------------|------|
| `board_type` | String | 게시판 유형 |
| `post_id` | String | 게시글 고유 ID |
| `author` | String | 작성자 |
| `publish_date` | String | 작성일 |
| `view_count` | Integer | 조회수 |
| `attachments` | Array | 첨부파일 목록 |
| `category` | String | 카테고리 |

#### 첨부파일 구조
```json
{
  "filename": "파일명.pdf",
  "url": "다운로드 URL",
  "size": "파일 크기"
}
```

## 데이터 품질 지표

### 1. 완성도 (Completeness)
- 필수 필드 누락률: < 1%
- 선택 필드 채움률: > 80%

### 2. 정확성 (Accuracy)
- URL 유효성: > 95%
- 감성 분석 정확도: > 85%

### 3. 일관성 (Consistency)
- 데이터 형식 표준화: 100%
- 타임스탬프 형식 통일: ISO 8601

### 4. 적시성 (Timeliness)
- 실시간 수집: 5분 이내
- 배치 처리: 일 1회

## 데이터 저장 형식

### 1. 원본 데이터
- **형식**: JSON
- **인코딩**: UTF-8
- **압축**: Gzip (선택사항)
- **저장 위치**: `data/raw/`

### 2. 전처리 데이터
- **형식**: JSON, CSV
- **인코딩**: UTF-8
- **저장 위치**: `data/processed/`

### 3. 분석 결과
- **형식**: JSON, Excel
- **인코딩**: UTF-8
- **저장 위치**: `data/analysis/`

## 데이터 수집 주기

| 플랫폼 | 수집 주기 | 최대 수집량 |
|--------|-----------|-------------|
| 네이버 | 실시간 | 100건/쿼리 |
| 유튜브 | 실시간 | 50건/쿼리 |
| 구글 | 실시간 | 100건/쿼리 |
| DCinside | 실시간 | 50건/쿼리 |
| 부안군청 | 일 1회 | 100건/수집 |

## 데이터 보안 및 개인정보

### 1. 개인정보 처리
- 개인정보 필터링 적용
- 익명화 처리
- 접근 권한 관리

### 2. 데이터 암호화
- 저장 시 암호화
- 전송 시 SSL/TLS
- 백업 데이터 암호화

### 3. 접근 제어
- 역할 기반 접근 제어
- 로그 기록
- 정기 보안 감사

## 데이터 백업 및 복구

### 1. 백업 정책
- 일일 증분 백업
- 주간 전체 백업
- 월간 아카이브

### 2. 복구 절차
- 데이터 복구 계획
- 복구 시간 목표: 4시간
- 복구 지점 목표: 1시간

## 데이터 아카이빙

### 1. 아카이빙 정책
- 1년 이상 된 데이터: 아카이브
- 3년 이상 된 데이터: 삭제
- 중요 데이터: 영구 보관

### 2. 아카이빙 형식
- 압축 형식: Gzip
- 메타데이터 포함
- 검색 가능한 인덱스

---

**문서 버전**: 1.0  
**최종 업데이트**: 2025년 월  
**작성자**: Stay_Buan 팀 