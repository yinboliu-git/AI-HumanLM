#!/usr/bin/env python3
"""
Moltbook Data Scraper
Scrapes conversation data from moltbook.com for research purposes
Designed to collect English-language posts, comments, and metadata for HuggingFace dataset
"""

import requests
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging
from pathlib import Path
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MoltbookScraper:
    """Scraper for Moltbook AI agent social network"""

    def __init__(self, output_dir: str = "moltbook_data", rate_limit: float = 0.5):
        """
        Initialize scraper

        Args:
            output_dir: Directory to save scraped data
            rate_limit: Seconds to wait between requests (default 0.5s)
        """
        self.base_url = "https://www.moltbook.com"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.rate_limit = rate_limit

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }

        # Statistics
        self.stats = {
            'posts_scraped': 0,
            'comments_scraped': 0,
            'submolts_scraped': 0,
            'errors': 0,
            'start_time': datetime.now()
        }

    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make API request with rate limiting and error handling"""
        url = f"{self.base_url}{endpoint}"

        try:
            time.sleep(self.rate_limit)  # Rate limiting
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            self.stats['errors'] += 1
            return None

    def is_english(self, text: str) -> bool:
        """
        Detect if text is primarily English
        Uses simple heuristic: >90% ASCII characters
        """
        if not text or text is None:
            return True
        ascii_ratio = sum(ord(c) < 128 for c in text) / len(text)
        return ascii_ratio > 0.9

    def scrape_submolts(self) -> List[Dict]:
        """Scrape all submolts (communities)"""
        logger.info("Scraping submolts...")
        submolts = []
        offset = 0
        limit = 100

        while True:
            data = self._request("/api/v1/submolts", params={'limit': limit, 'offset': offset})
            if not data or not data.get('success'):
                break

            batch = data.get('submolts', [])
            if not batch:
                break

            submolts.extend(batch)
            self.stats['submolts_scraped'] += len(batch)
            logger.info(f"Scraped {len(submolts)} submolts so far...")

            if not data.get('has_more', False):
                break

            offset = data.get('next_offset', offset + limit)

        # Save submolts
        output_file = self.output_dir / "submolts.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for submolt in submolts:
                f.write(json.dumps(submolt, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(submolts)} submolts to {output_file}")
        return submolts

    def scrape_posts(self, max_posts: Optional[int] = None, english_only: bool = True) -> List[Dict]:
        """
        Scrape posts from the main feed

        Args:
            max_posts: Maximum number of posts to scrape (None = all)
            english_only: Only scrape English posts
        """
        logger.info(f"Scraping posts (max: {max_posts or 'unlimited'}, english_only: {english_only})...")
        posts = []
        offset = 0
        limit = 100

        while True:
            if max_posts and len(posts) >= max_posts:
                break

            data = self._request("/api/v1/posts", params={'limit': limit, 'offset': offset})
            if not data or not data.get('success'):
                break

            batch = data.get('posts', [])
            if not batch:
                break

            # Filter for English if requested
            if english_only:
                batch = [p for p in batch if self.is_english(
                    (p.get('content') or '') + (p.get('title') or '')
                )]

            posts.extend(batch)
            self.stats['posts_scraped'] += len(batch)
            logger.info(f"Scraped {len(posts)} posts so far...")

            if not data.get('has_more', False):
                break

            offset = data.get('next_offset', offset + limit)

        # Save posts
        output_file = self.output_dir / "posts.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for post in posts:
                f.write(json.dumps(post, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(posts)} posts to {output_file}")
        return posts

    def scrape_post_with_comments(self, post_id: str) -> Optional[Dict]:
        """Scrape a single post with all its comments"""
        data = self._request(f"/api/v1/posts/{post_id}")
        if not data or not data.get('success'):
            return None

        post_data = {
            'post': data.get('post'),
            'comments': data.get('comments', []),
            'context': data.get('context')
        }

        self.stats['comments_scraped'] += len(post_data['comments'])
        return post_data

    def scrape_all_posts_with_comments(self, post_ids: List[str], english_only: bool = True) -> List[Dict]:
        """
        Scrape detailed data for multiple posts including comments

        Args:
            post_ids: List of post IDs to scrape
            english_only: Only include English comments
        """
        logger.info(f"Scraping {len(post_ids)} posts with comments...")
        detailed_posts = []

        for i, post_id in enumerate(post_ids):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(post_ids)} posts with comments...")

            post_data = self.scrape_post_with_comments(post_id)
            if not post_data:
                continue

            # Filter English comments if requested
            if english_only:
                post_data['comments'] = [
                    c for c in post_data['comments']
                    if self.is_english(c.get('content') or '')
                ]

            detailed_posts.append(post_data)

        # Save detailed posts
        output_file = self.output_dir / "posts_with_comments.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for post in detailed_posts:
                f.write(json.dumps(post, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(detailed_posts)} detailed posts to {output_file}")
        return detailed_posts

    def create_huggingface_dataset(self):
        """
        Convert scraped data to HuggingFace dataset format
        Creates a structured dataset with conversations
        """
        logger.info("Creating HuggingFace dataset format...")

        # Load posts with comments
        posts_file = self.output_dir / "posts_with_comments.jsonl"
        if not posts_file.exists():
            logger.error("No posts_with_comments.jsonl found. Run full scrape first.")
            return

        dataset = []
        with open(posts_file, 'r', encoding='utf-8') as f:
            for line in f:
                post_data = json.loads(line)
                post = post_data['post']
                comments = post_data['comments']

                # Create conversation format
                conversation = {
                    'id': post['id'],
                    'title': post.get('title', ''),
                    'post_content': post.get('content', ''),
                    'post_author': post.get('author', {}).get('name', ''),
                    'post_author_id': post.get('author', {}).get('id', ''),
                    'submolt': post.get('submolt', {}).get('name', ''),
                    'submolt_display_name': post.get('submolt', {}).get('display_name', ''),
                    'upvotes': post.get('upvotes', 0),
                    'downvotes': post.get('downvotes', 0),
                    'created_at': post.get('created_at', ''),
                    'url': post.get('url'),
                    'comment_count': len(comments),
                    'comments': [
                        {
                            'id': c.get('id', ''),
                            'content': c.get('content', ''),
                            'author': c.get('author', {}).get('name', ''),
                            'author_id': c.get('author', {}).get('id', ''),
                            'upvotes': c.get('upvotes', 0),
                            'downvotes': c.get('downvotes', 0),
                            'created_at': c.get('created_at', ''),
                            'parent_id': c.get('parent_id')
                        }
                        for c in comments
                    ]
                }
                dataset.append(conversation)

        # Save HuggingFace format
        output_file = self.output_dir / "moltbook_dataset.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"Created HuggingFace dataset with {len(dataset)} conversations")

        # Create metadata file
        metadata = {
            'dataset_name': 'moltbook_conversations',
            'description': 'Conversations from Moltbook - AI agent social network',
            'language': 'en',
            'total_conversations': len(dataset),
            'total_comments': sum(item['comment_count'] for item in dataset),
            'scraped_at': datetime.now().isoformat(),
            'source': 'https://www.moltbook.com',
            'license': 'Research purposes - check Moltbook ToS',
        }

        with open(self.output_dir / "dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Dataset metadata saved")
        return dataset

    def print_stats(self):
        """Print scraping statistics"""
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        logger.info("\n" + "="*50)
        logger.info("SCRAPING STATISTICS")
        logger.info("="*50)
        logger.info(f"Posts scraped: {self.stats['posts_scraped']}")
        logger.info(f"Comments scraped: {self.stats['comments_scraped']}")
        logger.info(f"Submolts scraped: {self.stats['submolts_scraped']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Time elapsed: {elapsed:.1f} seconds")
        logger.info(f"Average rate: {(self.stats['posts_scraped'] + self.stats['comments_scraped']) / max(elapsed, 1):.2f} items/sec")
        logger.info("="*50 + "\n")


def main():
    """Main scraping workflow"""
    scraper = MoltbookScraper(output_dir="moltbook_data", rate_limit=0.5)

    logger.info("Starting Moltbook scraper...")
    logger.info("Strategy: Scrape posts + detailed comments to achieve 90%+ coverage")

    # Step 1: Scrape submolts (communities)
    submolts = scraper.scrape_submolts()

    # Step 2: Scrape posts from main feed (English only)
    # The API seems to paginate with limits, let's scrape extensively
    posts = scraper.scrape_posts(max_posts=50000, english_only=True)

    # Step 3: Get detailed data with comments for all posts
    post_ids = [p['id'] for p in posts]
    detailed_posts = scraper.scrape_all_posts_with_comments(post_ids, english_only=True)

    # Step 4: Create HuggingFace dataset format
    scraper.create_huggingface_dataset()

    # Print final statistics
    scraper.print_stats()

    logger.info(f"All data saved to: {scraper.output_dir}")
    logger.info("Dataset ready for HuggingFace upload!")


if __name__ == "__main__":
    main()
