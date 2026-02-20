import random
import time
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Optional

from smolagents.tools import Tool
from smolagents import DuckDuckGoSearchTool, GoogleSearchTool


class RobustVisitWebpageTool(Tool):
    """Enhanced VisitWebpageTool with balanced robustness - avoids 403 errors while maintaining compatibility"""

    name = "visit_webpage"
    description = (
        "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }
    output_type = "string"

    def __init__(self, max_output_length: int = 40000, max_retries: int = 3,
                 db_path: str = "webpage_cache.db", cache_ttl_hours: int = 72):
        """Initialize with database caching support.

        Args:
            max_output_length: Maximum length of output content
            max_retries: Maximum number of retries for failed requests
            db_path: Path to SQLite database file for caching
            cache_ttl_hours: Time to live for cached results in hours (default 72 hours)
        """
        super().__init__()
        self.max_output_length = max_output_length
        self.max_retries = max_retries
        self.db_path = db_path
        self.cache_ttl_hours = cache_ttl_hours
        self._init_database()

        # Conservative user agents - avoid problematic ones
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        ]

    def _init_database(self):
        """Initialize the cache database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS webpage_cache (
                url_hash TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_webpage_created_at ON webpage_cache(created_at)
        """)
        conn.commit()
        conn.close()

    def _get_url_hash(self, url: str) -> str:
        """Generate a unique hash for the URL."""
        return hashlib.sha256(url.encode()).hexdigest()

    def _get_cached_content(self, url: str) -> Optional[str]:
        """Check if we have cached content for this URL."""
        url_hash = self._get_url_hash(url)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get cached content if it exists and is not expired
        cutoff_time = datetime.now() - timedelta(hours=self.cache_ttl_hours)
        cursor.execute("""
            SELECT content FROM webpage_cache
            WHERE url_hash = ? AND created_at > ?
        """, (url_hash, cutoff_time))

        result = cursor.fetchone()
        conn.close()

        if result:
            return result[0]
        return None

    def _cache_content(self, url: str, content: str):
        """Cache the webpage content."""
        url_hash = self._get_url_hash(url)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert or replace the cached content
        cursor.execute("""
            INSERT OR REPLACE INTO webpage_cache
            (url_hash, url, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (url_hash, url, content, datetime.now()))

        conn.commit()
        conn.close()


    def _get_headers(self):
        """Generate safe headers that work well across sites"""
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def _truncate_content(self, content: str, max_length: int) -> str:
        if len(content) <= max_length:
            return content
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2 :]
        )

    def forward(self, url: str) -> str:
        # Check if we have cached content for this URL
        cached_content = self._get_cached_content(url)
        if cached_content:
            print(f"[Cache Hit] Returning cached content for URL: {url[:100]}...")
            return cached_content

        print(f"[Cache Miss] Fetching new content for URL: {url[:100]}...")

        try:
            import re
            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException
        except ImportError as e:
            raise ImportError(
                "You must install packages `markdownify` and `requests` to run this tool: for instance run `pip install markdownify requests`."
            ) from e

        # Enhanced retry mechanism with headers rotation and fallback
        for attempt in range(self.max_retries + 1):
            try:
                # Add delay between retries with some randomness
                if attempt > 0:
                    time.sleep(random.uniform(2, 5))

                # Try with headers first, then fallback to no headers
                if attempt < self.max_retries:
                    headers = self._get_headers()
                    response = requests.get(url, headers=headers, timeout=20)
                else:
                    # Final attempt: use original approach (no custom headers)
                    response = requests.get(url, timeout=20)

                # Handle common anti-bot status codes
                if response.status_code == 403 and attempt < self.max_retries:
                    continue  # Retry with different approach

                response.raise_for_status()  # Raise an exception for bad status codes

                # Convert the HTML content to Markdown
                markdown_content = markdownify(response.text).strip()

                # Check for potential encoding issues (binary data in text)
                if len(markdown_content) > 100 and markdown_content.count('\x00') > 10:
                    if attempt < self.max_retries:
                        continue  # Retry, likely encoding issue

                # Remove multiple line breaks
                markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

                result = self._truncate_content(markdown_content, self.max_output_length)

                # Cache the successful result
                self._cache_content(url, result)

                return result

            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    continue
                return "The request timed out. Please try again later or check the URL."
            except RequestException as e:
                if attempt < self.max_retries:
                    continue
                return f"Error fetching the webpage: {str(e)}"
            except Exception as e:
                if attempt < self.max_retries:
                    continue
                return f"An unexpected error occurred: {str(e)}"

        return "Failed to fetch webpage after retries."


class MyGoogleSearchTool(DuckDuckGoSearchTool):
    name = "web_search"
    description = """Performs a google web search for your query then returns a string of the top search results."""
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."},
        "filter_year": {
            "type": "integer",
            "description": "Optionally restrict results to a certain year",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, *args, db_path: str = "search_cache.db", cache_ttl_hours: int = 24, **kwargs):
        """Initialize with database caching support.

        Args:
            db_path: Path to SQLite database file for caching
            cache_ttl_hours: Time to live for cached results in hours
        """
        super().__init__(*args, **kwargs)
        self.db_path = db_path
        self.cache_ttl_hours = cache_ttl_hours
        self._init_database()
        self.smolagents_google = GoogleSearchTool(provider="organic")

    def _init_database(self):
        """Initialize the cache database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                query_hash TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                results TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON search_cache(created_at)
        """)
        conn.commit()
        conn.close()

    def _get_query_hash(self, query: str) -> str:
        """Generate a unique hash for the query."""
        return hashlib.sha256(query.encode()).hexdigest()

    def _get_cached_result(self, query: str) -> Optional[str]:
        """Check if we have a cached result for this query."""
        query_hash = self._get_query_hash(query)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get cached result if it exists and is not expired
        cutoff_time = datetime.now() - timedelta(hours=self.cache_ttl_hours)
        cursor.execute("""
            SELECT results FROM search_cache
            WHERE query_hash = ? AND created_at > ?
        """, (query_hash, cutoff_time))

        result = cursor.fetchone()
        conn.close()

        if result:
            return result[0]
        return None

    def _cache_result(self, query: str, result: str):
        """Cache the search result."""
        query_hash = self._get_query_hash(query)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert or replace the cached result
        cursor.execute("""
            INSERT OR REPLACE INTO search_cache
            (query_hash, query, results, created_at)
            VALUES (?, ?, ?, ?)
        """, (query_hash, query, result, datetime.now()))

        conn.commit()
        conn.close()


    def forward(self, query: str, filter_year: int | None = None) -> str:
        # Check if we have a cached result (ignoring filter_year since it's not supported)
        cached_result = self._get_cached_result(query)
        if cached_result:
            print(f"[Cache Hit] Returning cached result for query: {query[:50]}...")
            # If filter_year was provided, add warning to cached result
            if filter_year is not None:
                return "filter_year is not supported in this tool.\n\n" + cached_result.replace("## Search Results", "").strip()
            return cached_result

        print(f"[Cache Miss] Performing new search for query: {query[:50]}...")

        # No cache found, perform the actual search
        self._enforce_rate_limit()
        try:
            results = self.ddgs.text(query, max_results=self.max_results, backend="google,mullvad_google")
        except:
            result_str = self.smolagents_google.forward(query)
            self._cache_result(query, result_str)
            return result_str
            # raise Exception("No results found! Try a less restrictive/shorter query.")

        web_snippets = []
        for idx, result in enumerate(results):
            web_snippets.append(f"{idx}. [{result['title']}]({result['href']})\n{result['body']}")

        # Always cache without the filter_year warning
        result_str = "## Search Results\n\n" + "\n\n".join(web_snippets)
        self._cache_result(query, result_str)

        # Return with warning if filter_year was provided
        if filter_year is not None:
            return "filter_year is not supported in this tool.\n\n" + result_str
        return result_str