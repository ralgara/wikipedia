"""Content filtering utilities for Wikipedia pageviews data."""

def is_content(article: str) -> bool:
    """Returns False for non-content pages (Main_Page, Special:*, etc.)

    Args:
        article: Article title to check

    Returns:
        True if the article is content, False otherwise
    """
    prefixes = (
        "Category:", "Draft:", "File:", "Help:", "Media:",
        "MediaWiki:", "Module:", "Portal:", "Special:",
        "Template:", "TimedText:", "User:", "Wikipedia:"
    )
    return not (
        article in ['Main_Page']
        or article.startswith(prefixes)
        or '_talk:' in article
    )


def should_flag_for_review(article: str) -> bool:
    """Returns True for articles that may need manual review.

    Conservative flagging of articles that might contain adult content.
    False positives are acceptable as they can be manually unhidden.

    Args:
        article: Article title to check

    Returns:
        True if article should be flagged for manual review
    """
    # Keywords that might indicate adult content
    review_keywords = [
        'pornography', 'pornographic', 'xxx', 'sexual_intercourse',
        'erotic', 'masturbation', 'porn_', '_porn', 'sex_position',
        'hentai', 'playboy', 'penthouse_(magazine)'
    ]
    article_lower = article.lower()
    return any(keyword in article_lower for keyword in review_keywords)


def get_hide_reason(article: str) -> str | None:
    """Returns reason to hide article, or None if should be visible.

    Args:
        article: Article title to check

    Returns:
        Hide reason string, or None if article should not be hidden
    """
    if not is_content(article):
        if article == 'Main_Page':
            return 'main_page'
        elif article.startswith('Special:'):
            return 'special_page'
        elif '_talk:' in article:
            return 'talk_page'
        else:
            return 'non_content_page'

    if should_flag_for_review(article):
        return 'flagged_for_review'

    return None
