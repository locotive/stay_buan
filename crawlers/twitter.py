import tweepy
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
            tweets = tweepy.Cursor(self.api.search_tweets, q=keyword, lang="ko").items(self.max_pages)
            
            for tweet in tweets:
                tweet_data = {
                    'text': tweet.text,
                    'user': tweet.user.screen_name,
                    'created_at': tweet.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count,
                    'url': f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}",
                    'platform': 'twitter',
                    'keyword': keyword,
                    'crawled_at': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                all_results.append(tweet_data)
            
            filename = self.generate_filename(keyword)
            DataSaver.save_json(all_results, filename, self.save_dir)
            self.logger.info(f"Saved {len(all_results)} results for keyword '{keyword}'")
        
        return all_results 