import json
import csv
import os
import pandas as pd
from datetime import datetime

class DataSaver:
    """데이터 저장 유틸리티"""
    
    @staticmethod
    def save_json(data, filename, save_dir="data/raw"):
        """JSON 형식으로 저장"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        filepath = os.path.join(save_dir, f"{filename}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return filepath
    
    @staticmethod
    def save_csv(data, filename, save_dir="data/raw"):
        """CSV 형식으로 저장"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        filepath = os.path.join(save_dir, f"{filename}.csv")
        
        # 리스트가 비어있으면 빈 CSV 생성
        if not data:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['empty_data'])
            return filepath
            
        # 판다스 데이터프레임으로 변환 후 저장
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        return filepath 