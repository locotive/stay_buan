import tweepy
import time
from urllib.parse import quote
from core.base_crawler import BaseCrawler
from utils.savers import DataSaver

class TwitterCrawler(BaseCrawler):
    """트위터 크롤러"""
    
    def __init__(self, keywords, max_results=100, save_dir="data/raw"):
        super().__init__(keywords, max_results, save_dir)
        self.api_key = "YOUR_TWITTER_API_KEY"
        self.api_secret = "YOUR_TWITTER_API_SECRET"
        self.access_token = "YOUR_TWITTER_ACCESS_TOKEN"
        self.access_token_secret = "YOUR_TWITTER_ACCESS_TOKEN_SECRET"
        
        # Tweepy API 설정
        auth = tweepy.OAuth1UserHandler(
            self.api_key, self.api_secret,
            self.access_token, self.access_token_secret
        )
        self.api = tweepy.API(auth)
    
    def crawl(self):
        """트위터 데이터 수집"""
        all_results = []
        
        for keyword in self.keywords:
            self.logger.info(f"Crawling Twitter for keyword: {keyword}")
            encoded_keyword = quote(keyword)
            keyword_results = []
            
            tweets = tweepy.Cursor(self.api.search_tweets, q=encoded_keyword, lang="ko").items(self.max_results)
            
            for tweet in tweets:
                tweet_data = {
                    'content': tweet.text,
                    'url': f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}",
                    'user': tweet.user.screen_name,
                    'published_date': tweet.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    'platform': 'twitter',
                    'keyword': keyword,
                    'crawled_at': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                keyword_results.append(tweet_data)
            
            # keyword별로 저장
            if keyword_results:
                filename = self.generate_filename(keyword)
                DataSaver.save_json(keyword_results, filename, self.save_dir)
                self.logger.info(f"Saved {len(keyword_results)} results for keyword '{keyword}'")
        
        return all_results 