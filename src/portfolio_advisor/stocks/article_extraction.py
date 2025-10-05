"""Service for extracting text from HTML news articles using Ollama."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..services.ollama_service import OllamaService
from ..utils.fs import utcnow_iso, write_json_atomic
from .db import StockPaths, _read_json

logger = logging.getLogger(__name__)


def clean_html(html: str) -> str:
    """Remove scripts, styles, and other noise from HTML.

    Based on ReaderLM-v2 recommendations for pre-processing.
    """
    # Patterns from ReaderLM-v2 documentation
    SCRIPT_PATTERN = r"<[ ]*script.*?\/[ ]*script[ ]*>"
    STYLE_PATTERN = r"<[ ]*style.*?\/[ ]*style[ ]*>"
    META_PATTERN = r"<[ ]*meta.*?>"
    COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
    LINK_PATTERN = r"<[ ]*link.*?>"
    BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
    SVG_PATTERN = r"(<svg[^>]*>)(.*?)(<\/svg>)"

    # Remove scripts
    html = re.sub(SCRIPT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    # Remove styles
    html = re.sub(STYLE_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    # Remove meta tags
    html = re.sub(META_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    # Remove comments
    html = re.sub(COMMENT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    # Remove link tags
    html = re.sub(LINK_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    # Remove base64 images
    html = re.sub(BASE64_IMG_PATTERN, '<img src="#"/>', html)
    # Replace SVG content with placeholder
    html = re.sub(
        SVG_PATTERN,
        lambda match: f"{match.group(1)}[SVG placeholder]{match.group(3)}",
        html,
        flags=re.DOTALL,
    )

    return html


def extract_article_section(html: str) -> str:
    """Try to extract just the article content section from HTML.

    This helps focus the model on the main content.
    """
    # For GlobeNewswire, look for the main-body-container with article-body class
    # This is very specific to avoid catching related content
    main_body_match = re.search(
        r'<div[^>]*class="[^"]*article-body[^"]*"[^>]*id="main-body-container"[^>]*>(.*?)</div>\s*<div[^>]*class="[^"]*(?:main-tags|related)',
        html,
        re.DOTALL | re.IGNORECASE,
    )
    if main_body_match:
        return main_body_match.group(0)

    # Fallback: Look for article body with itemprop
    itemprop_match = re.search(
        r'<div[^>]*itemprop="articleBody"[^>]*>(.*?)(?=<div[^>]*class="[^"]*(?:related|tags|footer))',
        html,
        re.DOTALL | re.IGNORECASE,
    )
    if itemprop_match:
        return itemprop_match.group(0)

    # If no specific article section found, return original HTML
    # Let the model figure it out with better prompting
    return html


class ArticleTextExtractionService:
    """Service for extracting text from HTML articles using LLM."""

    # Prompt template for article extraction - optimized for verbatim text
    EXTRACTION_PROMPT = """Extract the main news article content from this HTML.

CRITICAL: Look for the FIRST major content section that contains news article paragraphs.
The article usually starts with location/date like "New York, USA, Sept. 25, 2025 (GLOBE NEWSWIRE)"

RULES:
1. Extract the complete article starting from the headline/title
2. Include all paragraphs, bullet points, and data from the main article
3. Output text EXACTLY as written - preserve all facts, numbers, and quotes
4. STOP when you reach: "Tags", "Related Links", "About", or similar footer sections
5. Do NOT include navigation, ads, or content from other articles

Output the full article text, starting with the headline.

HTML:
{html_content}"""

    def __init__(
        self,
        paths: StockPaths,
        ollama_service: OllamaService,
        model: str = "milkey/reader-lm-v2:Q8_0",
    ):
        """Initialize with stock paths and ollama service.

        Args:
            paths: Stock paths configuration
            ollama_service: Ollama service instance
            model: Model name to use for extraction
        """
        self.paths = paths
        self.ollama = ollama_service
        self.model = model

    def extract_article_text(
        self, ticker_slug: str, article_id: str, force: bool = False
    ) -> dict[str, Any]:
        """Extract text from a single article.

        Args:
            ticker_slug: Ticker slug identifier
            article_id: Article ID
            force: Force re-extraction even if text already exists

        Returns:
            Dictionary with extraction statistics
        """
        stats = {
            "article_id": article_id,
            "success": False,
            "skipped": False,
            "error": None,
            "extracted_chars": 0,
        }

        try:
            # Check if article metadata exists
            article_json_path = self.paths.news_article_json(ticker_slug, article_id)
            if not article_json_path.exists():
                stats["error"] = "Article metadata not found"
                return stats

            article_data = _read_json(article_json_path)

            # Check if already extracted and not forcing
            if not force and article_data.get("text_extracted"):
                stats["skipped"] = True
                stats["success"] = True
                return stats

            # Find HTML file
            local_content = article_data.get("local_content", {})
            content_path = local_content.get("content_path")
            if not content_path:
                stats["error"] = "No local content path"
                return stats

            html_path = self.paths.news_dir(ticker_slug) / content_path
            if not html_path.exists():
                stats["error"] = f"HTML file not found: {content_path}"
                return stats

            # Read HTML content
            html_content = html_path.read_text(encoding="utf-8", errors="ignore")

            # Extract article section first to focus on main content
            html_content = extract_article_section(html_content)

            # Clean HTML to remove noise
            html_content = clean_html(html_content)

            # Truncate very large HTML files to avoid token limits
            max_html_chars = 100000  # Increased limit since we're cleaning HTML
            if len(html_content) > max_html_chars:
                logger.warning(
                    f"Truncating HTML for {article_id} from {len(html_content)} "
                    f"to {max_html_chars} chars"
                )
                html_content = html_content[:max_html_chars]

            # Extract text using ollama
            prompt = self.EXTRACTION_PROMPT.format(html_content=html_content)

            # System prompt for better guidance
            system_prompt = (
                "You are a news article extraction tool. Your job is to extract the main news "
                "article content from HTML pages. Focus on the primary article, not supplementary "
                "content. Extract the complete text verbatim without summarizing."
            )

            extracted_text = self.ollama.generate(
                model=self.model,
                prompt=prompt,
                system=system_prompt,
                temperature=0.0,  # Zero temperature for deterministic extraction
                max_tokens=8000,  # Increased limit for full article text
            )

            # Save extracted text
            text_path = self.paths.news_article_content(ticker_slug, article_id, "txt")
            text_path.write_text(extracted_text, encoding="utf-8")

            # Update article metadata
            article_data["text_extracted"] = True
            article_data["text_extracted_at"] = utcnow_iso()
            article_data["text_extraction_model"] = self.model
            article_data["text_content"] = {
                "content_path": f"articles/{text_path.name}",
                "content_size_bytes": len(extracted_text.encode("utf-8")),
                "content_type": "text/plain",
            }
            write_json_atomic(article_json_path, article_data)

            stats["success"] = True
            stats["extracted_chars"] = len(extracted_text)

            logger.debug(f"Extracted {len(extracted_text)} chars from {article_id}")

        except Exception as e:
            logger.error(f"Error extracting text from {article_id}: {e}")
            stats["error"] = str(e)

        return stats

    def extract_all_articles(
        self, ticker_slug: str, force: bool = False, batch_size: int = 10
    ) -> dict[str, Any]:
        """Extract text from all articles for a ticker.

        Args:
            ticker_slug: Ticker slug identifier
            force: Force re-extraction even if text already exists
            batch_size: Number of articles to process in parallel

        Returns:
            Dictionary with extraction statistics
        """
        stats = {
            "ticker_slug": ticker_slug,
            "total_articles": 0,
            "extracted": 0,
            "skipped": 0,
            "errors": 0,
            "details": [],
        }

        # Load news index
        index_path = self.paths.news_index_json(ticker_slug)
        if not index_path.exists():
            logger.warning(f"No news index found for {ticker_slug}")
            return stats

        index = _read_json(index_path)
        articles = index.get("articles", {})
        stats["total_articles"] = len(articles)

        if not articles:
            return stats

        # Process articles in batches
        article_ids = list(articles.keys())
        for i in range(0, len(article_ids), batch_size):
            batch = article_ids[i : i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}/"
                f"{(len(article_ids) + batch_size - 1)//batch_size} "
                f"for {ticker_slug}"
            )

            for article_id in batch:
                result = self.extract_article_text(ticker_slug, article_id, force=force)
                stats["details"].append(result)

                if result["success"]:
                    if result["skipped"]:
                        stats["skipped"] += 1
                    else:
                        stats["extracted"] += 1
                else:
                    stats["errors"] += 1

        # Update news index with extraction status
        for article_id, article_info in articles.items():
            article_json_path = self.paths.news_article_json(ticker_slug, article_id)
            if article_json_path.exists():
                article_data = _read_json(article_json_path)
                if article_data.get("text_extracted"):
                    article_info["text_extracted"] = True
                    article_info["text_extracted_at"] = article_data.get("text_extracted_at")
                    article_info["text_extraction_model"] = article_data.get(
                        "text_extraction_model"
                    )

        index["last_extraction_run"] = utcnow_iso()
        write_json_atomic(index_path, index)

        return stats

    def extract_portfolio_articles(
        self, portfolio_path: Path, force: bool = False, max_workers: int = 4
    ) -> dict[str, Any]:
        """Extract text from all articles in portfolio.

        Args:
            portfolio_path: Path to portfolio output directory
            force: Force re-extraction even if text already exists
            max_workers: Maximum parallel workers

        Returns:
            Dictionary with extraction statistics
        """
        stats = {
            "total_tickers": 0,
            "processed_tickers": 0,
            "total_articles": 0,
            "extracted": 0,
            "skipped": 0,
            "errors": 0,
            "ticker_details": {},
        }

        # Find all tickers with news
        stocks_dir = portfolio_path / "stocks" / "tickers"
        if not stocks_dir.exists():
            logger.warning(f"No stocks directory found at {stocks_dir}")
            return stats

        ticker_dirs = [d for d in stocks_dir.iterdir() if d.is_dir()]
        stats["total_tickers"] = len(ticker_dirs)

        # Check ollama availability
        if not self.ollama.is_available():
            logger.error("Ollama service is not available")
            stats["errors"] = stats["total_tickers"]
            return stats

        # Check if model exists
        if not self.ollama.model_exists(self.model):
            logger.error(f"Model {self.model} not found. Please run: ollama pull {self.model}")
            stats["errors"] = stats["total_tickers"]
            return stats

        # Process tickers in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {}

            for ticker_dir in ticker_dirs:
                ticker_slug = ticker_dir.name
                news_dir = ticker_dir / "primary" / "news"
                if news_dir.exists():
                    future = executor.submit(self.extract_all_articles, ticker_slug, force=force)
                    future_to_ticker[future] = ticker_slug

            for future in as_completed(future_to_ticker):
                ticker_slug = future_to_ticker[future]
                try:
                    ticker_stats = future.result()
                    stats["processed_tickers"] += 1
                    stats["total_articles"] += ticker_stats["total_articles"]
                    stats["extracted"] += ticker_stats["extracted"]
                    stats["skipped"] += ticker_stats["skipped"]
                    stats["errors"] += ticker_stats["errors"]
                    stats["ticker_details"][ticker_slug] = ticker_stats

                    logger.info(
                        f"Processed {ticker_slug}: "
                        f"{ticker_stats['extracted']} extracted, "
                        f"{ticker_stats['skipped']} skipped, "
                        f"{ticker_stats['errors']} errors"
                    )
                except Exception as e:
                    logger.error(f"Error processing {ticker_slug}: {e}")
                    stats["errors"] += 1

        return stats
