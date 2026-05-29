# AI 서버 (FastAPI)

> 실행 순서: **인프라 → 백엔드 → AI서버 → 목서버**
>
> 인프라(Docker)가 먼저 실행된 상태여야 합니다.

## 실행

```powershell
# c:\itstudy\ai-server

# venv 활성화 (터미널 새로 열 때마다)
.\.venv\Scripts\activate

# 실행
uvicorn app.main:app --reload --port 8000
```

## 포트

| 서비스    | 포트 |
|-----------|------|
| AI Server | 8000 |
