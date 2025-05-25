import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)

class ModelCache:
    """모델 캐시 관리 클래스"""
    
    def __init__(self, cache_dir="data/models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 설정
        self.model_configs = {
            'kobert': {
                'name': 'skt/kobert-base-v1',
                'local_path': self.cache_dir / 'kobert'
            },
            'kcbert': {
                'name': 'beomi/kcbert-base',
                'local_path': self.cache_dir / 'kcbert'
            },
            'koalpaca': {
                'name': 'beomi/KoAlpaca-Polyglot-12.8B',
                'local_path': self.cache_dir / 'koalpaca'
            }
        }
        
        # 모델 인스턴스 캐시
        self._model_instances = {}
        self._tokenizer_instances = {}
    
    def get_model(self, model_name: str):
        """모델 인스턴스 반환 (캐시된 인스턴스 재사용)"""
        if model_name not in self._model_instances:
            config = self.model_configs[model_name]
            try:
                # 로컬 모델 파일 확인
                if not config['local_path'].exists():
                    logger.warning(f"{model_name} 모델이 로컬에 없습니다. 온라인 다운로드를 시도합니다.")
                    return None
                
                # 모델 로드
                self._model_instances[model_name] = AutoModelForSequenceClassification.from_pretrained(
                    config['local_path'],
                    local_files_only=True
                )
                
            except Exception as e:
                logger.error(f"{model_name} 모델 로드 실패: {str(e)}")
                return None
                
        return self._model_instances[model_name]
    
    def get_tokenizer(self, model_name: str):
        """토크나이저 인스턴스 반환 (캐시된 인스턴스 재사용)"""
        if model_name not in self._tokenizer_instances:
            config = self.model_configs[model_name]
            try:
                # 로컬 모델 파일 확인
                if not config['local_path'].exists():
                    logger.warning(f"{model_name} 토크나이저가 로컬에 없습니다. 온라인 다운로드를 시도합니다.")
                    return None
                
                # 토크나이저 로드
                self._tokenizer_instances[model_name] = AutoTokenizer.from_pretrained(
                    config['local_path'],
                    local_files_only=True
                )
                
            except Exception as e:
                logger.error(f"{model_name} 토크나이저 로드 실패: {str(e)}")
                return None
                
        return self._tokenizer_instances[model_name]
