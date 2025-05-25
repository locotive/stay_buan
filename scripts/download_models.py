from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_name, repo_id, local_dir):
    """모델 다운로드"""
    try:
        logger.info(f"{model_name} 모델 다운로드 시작...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        logger.info(f"{model_name} 모델 다운로드 완료")
    except Exception as e:
        logger.error(f"{model_name} 모델 다운로드 실패: {str(e)}")

def main():
    models = {
        'kobert': 'skt/kobert-base-v1',
        'kcbert': 'beomi/kcbert-base',
        'koalpaca': 'beomi/KoAlpaca-Polyglot-12.8B'
    }
    
    base_dir = "data/models"
    os.makedirs(base_dir, exist_ok=True)
    
    for name, repo_id in models.items():
        local_dir = os.path.join(base_dir, name)
        download_model(name, repo_id, local_dir)

if __name__ == "__main__":
    main()
