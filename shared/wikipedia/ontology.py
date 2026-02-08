"""Ontology integration module for Wikipedia article enrichment.

This module provides clients for querying Wikidata and DBpedia to enrich
Wikipedia article metadata with semantic relationships, categories, and entity types.

Usage:
    from shared.wikipedia.ontology import WikidataClient, DBpediaClient
    
    wikidata = WikidataClient()
    metadata = wikidata.get_entity_metadata("Elizabeth_II")
    similarity = wikidata.semantic_similarity("Elizabeth_II", "Queen_Victoria")
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import quote

import requests
import requests_cache
from SPARQLWrapper import SPARQLWrapper, JSON

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = Path.home() / '.cache' / 'wikipedia_ontology'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DB = CACHE_DIR / 'requests_cache.db'
CACHE_TTL = 30 * 24 * 60 * 60  # 30 days

# Initialize requests cache
requests_cache.install_cache(
    str(CACHE_DB),
    backend='sqlite',
    expire_after=CACHE_TTL
)


class WikidataClient:
    """Client for querying Wikidata API.
    
    Provides methods to:
    - Link Wikipedia article names to Wikidata entity IDs
    - Fetch entity metadata (types, categories, properties)
    - Calculate semantic similarity between entities
    """
    
    BASE_URL = "https://www.wikidata.org/w/api.php"
    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    
    # User-Agent required by Wikimedia API policy
    USER_AGENT = "WikipediaPageviewsAnalytics/1.0 (https://github.com/wikipedia-analytics; contact@example.com) Python-requests"
    
    def __init__(self, cache_ttl: int = CACHE_TTL):
        """Initialize Wikidata client.
        
        Args:
            cache_ttl: Cache time-to-live in seconds (default: 30 days)
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.USER_AGENT
        })
        self.cache_ttl = cache_ttl
        
    def _make_request(self, params: Dict) -> Dict:
        """Make API request with caching and error handling.
        
        Args:
            params: API request parameters
            
        Returns:
            JSON response data
        """
        params['format'] = 'json'
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Wikidata API request failed: {e}")
            return {}
    
    def get_entity_id(self, article: str) -> Optional[str]:
        """Get Wikidata entity ID for Wikipedia article.
        
        Args:
            article: Wikipedia article name (e.g., "Elizabeth_II")
            
        Returns:
            Wikidata entity ID (e.g., "Q9682") or None if not found
        """
        # Convert underscores to spaces for Wikipedia title
        title = article.replace('_', ' ')
        
        params = {
            'action': 'wbgetentities',
            'sites': 'enwiki',
            'titles': title,
            'props': 'info'
        }
        
        data = self._make_request(params)
        if not data or 'entities' not in data:
            return None
        
        # Extract entity ID from response
        for entity_id, entity_data in data['entities'].items():
            if entity_id != '-1':  # -1 indicates not found
                return entity_id
        
        return None
    
    def get_entity_data(self, entity_id: str) -> Dict:
        """Get full entity data from Wikidata.
        
        Args:
            entity_id: Wikidata entity ID (e.g., "Q9682")
            
        Returns:
            Dictionary containing entity data
        """
        params = {
            'action': 'wbgetentities',
            'ids': entity_id,
            'props': 'labels|claims|descriptions',
            'languages': 'en'
        }
        
        data = self._make_request(params)
        if not data or 'entities' not in data:
            return {}
        
        return data['entities'].get(entity_id, {})
    
    def get_entity_metadata(self, article: str) -> Dict:
        """Get enriched metadata for Wikipedia article.
        
        Args:
            article: Wikipedia article name
            
        Returns:
            Dictionary with structure:
            {
                'wikidata_id': str,
                'entity_type': str,  # P31 instance of
                'entity_type_label': str,
                'categories': List[str],  # P31 and P279 labels
                'related_entities': Dict[str, List[str]],  # Property -> entity IDs
                'description': str
            }
        """
        entity_id = self.get_entity_id(article)
        if not entity_id:
            logger.warning(f"No Wikidata entity found for {article}")
            return {}
        
        entity_data = self.get_entity_data(entity_id)
        if not entity_data:
            return {}
        
        # Extract claims (properties and values)
        claims = entity_data.get('claims', {})
        
        # Get entity type (P31 - instance of)
        entity_types = []
        entity_type_labels = []
        if 'P31' in claims:
            for claim in claims['P31'][:3]:  # Limit to first 3 types
                if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                    value = claim['mainsnak']['datavalue'].get('value', {})
                    if 'id' in value:
                        entity_types.append(value['id'])
                        # Get label for this type
                        type_data = self.get_entity_data(value['id'])
                        if 'labels' in type_data and 'en' in type_data['labels']:
                            entity_type_labels.append(type_data['labels']['en']['value'])
        
        # Get categories/subclasses (P279 - subclass of)
        categories = list(entity_type_labels)  # Start with entity types
        if 'P279' in claims:
            for claim in claims['P279'][:5]:
                if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                    value = claim['mainsnak']['datavalue'].get('value', {})
                    if 'id' in value:
                        cat_data = self.get_entity_data(value['id'])
                        if 'labels' in cat_data and 'en' in cat_data['labels']:
                            categories.append(cat_data['labels']['en']['value'])
        
        # Get related entities (common properties)
        related_props = {
            'P22': 'father',
            'P25': 'mother',
            'P26': 'spouse',
            'P40': 'children',
            'P3373': 'siblings',
            'P39': 'position_held',
            'P108': 'employer',
            'P69': 'educated_at'
        }
        
        related_entities = {}
        for prop_id, prop_name in related_props.items():
            if prop_id in claims:
                entities = []
                for claim in claims[prop_id]:
                    if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                        value = claim['mainsnak']['datavalue'].get('value', {})
                        if 'id' in value:
                            entities.append(value['id'])
                if entities:
                    related_entities[prop_id] = entities
        
        # Get description
        description = ""
        if 'descriptions' in entity_data and 'en' in entity_data['descriptions']:
            description = entity_data['descriptions']['en']['value']
        
        return {
            'wikidata_id': entity_id,
            'entity_type': entity_types[0] if entity_types else None,
            'entity_type_label': entity_type_labels[0] if entity_type_labels else None,
            'categories': categories,
            'related_entities': related_entities,
            'description': description
        }
    
    def semantic_similarity(self, article1: str, article2: str) -> float:
        """Calculate semantic similarity between two articles.
        
        Uses category overlap (Jaccard similarity) as a simple metric.
        Future: Could compute ontology distance in taxonomy tree.
        
        Args:
            article1: First Wikipedia article name
            article2: Second Wikipedia article name
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        meta1 = self.get_entity_metadata(article1)
        meta2 = self.get_entity_metadata(article2)
        
        if not meta1 or not meta2:
            return 0.0
        
        # Jaccard similarity of categories
        cats1 = set(meta1.get('categories', []))
        cats2 = set(meta2.get('categories', []))
        
        if not cats1 or not cats2:
            return 0.0
        
        intersection = len(cats1 & cats2)
        union = len(cats1 | cats2)
        
        return intersection / union if union > 0 else 0.0


class DBpediaClient:
    """Client for querying DBpedia SPARQL endpoint.
    
    Provides complementary data to Wikidata, particularly for categories
    and ontology class membership.
    """
    
    SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
    USER_AGENT = "WikipediaPageviewsAnalytics/1.0 (https://github.com/wikipedia-analytics; contact@example.com) SPARQLWrapper"
    
    def __init__(self):
        """Initialize DBpedia SPARQL client."""
        self.sparql = SPARQLWrapper(self.SPARQL_ENDPOINT)
        self.sparql.setReturnFormat(JSON)
        self.sparql.addCustomHttpHeader('User-Agent', self.USER_AGENT)
    
    def _execute_query(self, query: str) -> List[Dict]:
        """Execute SPARQL query with error handling.
        
        Args:
            query: SPARQL query string
            
        Returns:
            List of result bindings
        """
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            return results['results']['bindings']
        except Exception as e:
            logger.error(f"DBpedia SPARQL query failed: {e}")
            return []
    
    def get_categories(self, article: str) -> List[str]:
        """Get DBpedia categories for Wikipedia article.
        
        Args:
            article: Wikipedia article name
            
        Returns:
            List of category names
        """
        # DBpedia resource URI
        resource = f"http://dbpedia.org/resource/{article}"
        
        query = f"""
        PREFIX dbc: <http://dbpedia.org/resource/Category:>
        PREFIX dct: <http://purl.org/dc/terms/>
        
        SELECT DISTINCT ?category WHERE {{
            <{resource}> dct:subject ?category .
        }}
        LIMIT 50
        """
        
        results = self._execute_query(query)
        
        categories = []
        for result in results:
            cat_uri = result['category']['value']
            # Extract category name from URI
            if 'Category:' in cat_uri:
                cat_name = cat_uri.split('Category:')[-1]
                categories.append(cat_name.replace('_', ' '))
        
        return categories
    
    def get_ontology_types(self, article: str) -> List[str]:
        """Get DBpedia ontology types for article.
        
        Args:
            article: Wikipedia article name
            
        Returns:
            List of ontology class URIs (e.g., dbo:Person)
        """
        resource = f"http://dbpedia.org/resource/{article}"
        
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        
        SELECT DISTINCT ?type WHERE {{
            <{resource}> rdf:type ?type .
            FILTER (STRSTARTS(STR(?type), "http://dbpedia.org/ontology/"))
        }}
        """
        
        results = self._execute_query(query)
        
        types = []
        for result in results:
            type_uri = result['type']['value']
            # Extract class name
            if '/ontology/' in type_uri:
                class_name = type_uri.split('/ontology/')[-1]
                types.append(class_name)
        
        return types


def get_enriched_metadata(article: str) -> Dict:
    """Get combined enriched metadata from both Wikidata and DBpedia.
    
    This is a convenience function that combines data from both sources.
    
    Args:
        article: Wikipedia article name
        
    Returns:
        Dictionary combining Wikidata and DBpedia metadata
    """
    wikidata = WikidataClient()
    dbpedia = DBpediaClient()
    
    # Get Wikidata metadata
    metadata = wikidata.get_entity_metadata(article)
    
    if metadata:
        # Enrich with DBpedia data
        db_categories = dbpedia.get_categories(article)
        db_types = dbpedia.get_ontology_types(article)
        
        metadata['dbpedia_categories'] = db_categories
        metadata['dbpedia_types'] = db_types
    
    return metadata
