"""Content filtering utilities for Wikipedia pageviews data."""

# Non-content namespaces in English Wikipedia
# See: https://en.wikipedia.org/wiki/Wikipedia:Namespace
NON_CONTENT_PREFIXES = (
    "Category:", "Draft:", "File:", "Help:", "Media:",
    "MediaWiki:", "Module:", "Portal:", "Special:",
    "Template:", "TimedText:", "User:", "Wikipedia:",
    "Gadget:", "Gadget_definition:"
)

TALK_PREFIXES = (
    "Talk:", "User_talk:", "Wikipedia_talk:", "File_talk:",
    "MediaWiki_talk:", "Template_talk:", "Help_talk:",
    "Category_talk:", "Portal_talk:", "Draft_talk:",
    "Module_talk:", "Gadget_talk:", "Gadget_definition_talk:"
)

ALL_NON_CONTENT_PREFIXES = NON_CONTENT_PREFIXES + TALK_PREFIXES


def is_content(article: str) -> bool:
    """Returns False for non-content pages (Main_Page, Special:*, etc.)

    Args:
        article: Article title to check

    Returns:
        True if the article is content, False otherwise
    """
    return not (
        article == 'Main_Page'
        or article.startswith(ALL_NON_CONTENT_PREFIXES)
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
    if article == 'Main_Page':
        return 'main_page'

    if article.startswith('Special:'):
        return 'special_page'

    if article.startswith(TALK_PREFIXES) or '_talk:' in article:
        return 'talk_page'

    if article.startswith(NON_CONTENT_PREFIXES):
        return 'non_content_page'

    if should_flag_for_review(article):
        return 'flagged_for_review'

    return None
