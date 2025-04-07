import requests
from core.base_crawler import BaseCrawler
from utils.savers import DataSaver

class YouTubeCrawler(BaseCrawler):
    """유튜브 크롤러"""
    
    def __init__(self, keywords, max_results=50, save_dir="data/raw"):
        super().__init__(keywords, max_results, save_dir)
        self.api_key = "YOUR_YOUTUBE_API_KEY"
        self.base_url = "https://www.googleapis.com/youtube/v3"
    
    def crawl(self):
        """유튜브 데이터 수집"""
        all_results = []
        
        for keyword in self.keywords:
            self.logger.info(f"Crawling YouTube for keyword: {keyword}")
            search_url = f"{self.base_url}/search"
            params = {
                'part': 'snippet',
                'q': keyword,
                'type': 'video',
                'maxResults': self.max_pages,
                'key': self.api_key
            }
            
            response = requests.get(search_url, params=params)
            if response.status_code == 200:
                videos = response.json().get('items', [])
                for video in videos:
                    video_id = video['id']['videoId']
                    video_data = self.get_video_details(video_id)
                    if video_data:
                        all_results.append(video_data)
            
            filename = self.generate_filename(keyword)
            DataSaver.save_json(all_results, filename, self.save_dir)
            self.logger.info(f"Saved {len(all_results)} results for keyword '{keyword}'")
        
        return all_results
    
    def get_video_details(self, video_id):
        """비디오 세부 정보 및 댓글 수집"""
        video_url = f"{self.base_url}/videos"
        params = {
            'part': 'snippet,statistics',
            'id': video_id,
            'key': self.api_key
        }
        
        response = requests.get(video_url, params=params)
        if response.status_code == 200:
            video_info = response.json().get('items', [])[0]
            video_data = {
                'title': video_info['snippet']['title'],
                'description': video_info['snippet']['description'],
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'published_date': video_info['snippet']['publishedAt'],
                'view_count': video_info['statistics'].get('viewCount', 0),
                'like_count': video_info['statistics'].get('likeCount', 0),
                'comment_count': video_info['statistics'].get('commentCount', 0),
                'platform': 'youtube',
                'keyword': self.keywords,
                'crawled_at': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            return video_data
        return None 