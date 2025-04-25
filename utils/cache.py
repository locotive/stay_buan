import os
import json

class JsonCache:
    def __init__(self, cache_file="data/cache/visited_urls.json"):
        self.cache_file = cache_file
        self.visited = set()
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.visited = set(json.load(f))

    def exists(self, url):
        return url in self.visited

    def save(self, url):
        self.visited.add(url)
        self._write()

    def _write(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(list(self.visited), f, ensure_ascii=False, indent=2)
