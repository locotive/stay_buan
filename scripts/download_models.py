from huggingface_hub import snapshot_download
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_name, repo_id, local_dir):
    """모델 다운로드"""
    try:
        logger.info(f"[{model_name}] 모델 다운로드 시작... -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            revision="main"  # or specific version/tag if needed
        )
        logger.info(f"[{model_name}] 모델 다운로드 완료")
    except Exception as e:
        logger.error(f"[{model_name}] 모델 다운로드 실패: {str(e)}")

def main():
    # 모델 설정 업데이트 (KoAlpaca 제외, 새로운 모델 추가)
    models = {
        'kobert': 'skt/kobert-base-v1',
        'kcelectra-base-v2022': 'beomi/kcelectra-base-v2022',
        'kcelectra': 'beomi/kcelectra-base',
        'kcbert-large': 'beomi/kcbert-large',
        'kosentencebert': 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
    }

    base_dir = os.path.join("data", "models")
    os.makedirs(base_dir, exist_ok=True)

    for name, repo_id in models.items():
        local_dir = os.path.join(base_dir, name)
        os.makedirs(local_dir, exist_ok=True)
        download_model(name, repo_id, local_dir)

if __name__ == "__main__":
    main()
