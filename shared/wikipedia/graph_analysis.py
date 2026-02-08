"""Graph analysis module for Wikipedia article relationships.

This module provides NetworkX-based graph construction and analysis for article
relationships based on correlation, semantic similarity, and temporal patterns.

Usage:
    from shared.wikipedia.graph_analysis import ArticleGraph
    
    graph = ArticleGraph()
    graph.add_correlations(correlation_pairs, min_threshold=0.4)
    graph.load_enriched_metadata('data/enriched_metadata/')
    
    communities = graph.detect_communities()
    central = graph.get_central_articles(metric='pagerank', top_n=20)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
try:
    import community as community_louvain
    HAVE_LOUVAIN = True
except ImportError:
    HAVE_LOUVAIN = False
    logging.warning("python-louvain not installed. Community detection will be limited.")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArticleGraph:
    """Graph-based article relationship analysis.
    
    Builds a weighted graph where:
    - Nodes = Wikipedia articles
    - Edges = relationships (correlation, semantic similarity, etc.)
    - Edge attributes = weight, type, lag, metadata
    """
    
    def __init__(self):
        """Initialize empty article graph."""
        self.graph = nx.Graph()
        self.enriched_metadata = {}  # article -> metadata dict
        
    def add_correlations(self, pairs: List[Dict], min_threshold: float = 0.4) -> None:
        """Add correlation edges to graph.
        
        Args:
            pairs: List of dicts with keys: article1, article2, correlation, lag
            min_threshold: Minimum correlation to include
        """
        for pair in pairs:
            corr = pair.get('correlation', 0)
            if corr < min_threshold:
                continue
            
            article1 = pair['article1']
            article2 = pair['article2']
            
            # Add nodes if not present
            if not self.graph.has_node(article1):
                self.graph.add_node(article1)
            if not self.graph.has_node(article2):
                self.graph.add_node(article2)
            
            # Add edge with attributes
            self.graph.add_edge(
                article1,
                article2,
                weight=corr,
                type='correlation',
                lag=pair.get('lag', 0),
                spike_overlap=pair.get('spike_overlap', 0)
            )
        
        logger.info(f"Added {len(pairs)} correlation edges to graph")
        logger.info(f"Graph now has {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")
    
    def load_enriched_metadata(self, enriched_dir: str) -> None:
        """Load enriched metadata for articles in graph.
        
        Args:
            enriched_dir: Path to directory containing enriched JSON files
        """
        enriched_path = Path(enriched_dir)
        if not enriched_path.exists():
            logger.warning(f"Enriched metadata directory not found: {enriched_dir}")
            return
        
        loaded = 0
        for node in self.graph.nodes():
            json_file = enriched_path / f"{node}.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        self.enriched_metadata[node] = metadata
                        
                        # Add metadata as node attributes
                        self.graph.nodes[node]['wikidata_id'] = metadata.get('wikidata_id')
                        self.graph.nodes[node]['entity_type'] = metadata.get('entity_type_label')
                        self.graph.nodes[node]['categories'] = metadata.get('categories', [])
                        
                        loaded += 1
                except Exception as e:
                    logger.error(f"Failed to load metadata for {node}: {e}")
        
        logger.info(f"Loaded enriched metadata for {loaded}/{self.graph.number_of_nodes()} nodes")
    
    def add_semantic_edges(self, min_similarity: float = 0.6) -> None:
        """Add semantic similarity edges between articles.
        
        Requires enriched metadata to be loaded first.
        
        Args:
            min_similarity: Minimum Jaccard similarity to create edge
        """
        if not self.enriched_metadata:
            logger.warning("No enriched metadata loaded. Cannot add semantic edges.")
            return
        
        # Get articles with enriched data
        enriched_articles = list(self.enriched_metadata.keys())
        
        added = 0
        for i, article1 in enumerate(enriched_articles):
            meta1 = self.enriched_metadata[article1]
            cats1 = set(meta1.get('categories', []))
            
            if not cats1:
                continue
            
            for article2 in enriched_articles[i+1:]:
                # Skip if already has correlation edge
                if self.graph.has_edge(article1, article2):
                    continue
                
                meta2 = self.enriched_metadata[article2]
                cats2 = set(meta2.get('categories', []))
                
                if not cats2:
                    continue
                
                # Calculate Jaccard similarity
                intersection = len(cats1 & cats2)
                union = len(cats1 | cats2)
                similarity = intersection / union if union > 0 else 0.0
                
                if similarity >= min_similarity:
                    self.graph.add_edge(
                        article1,
                        article2,
                        weight=similarity,
                        type='semantic',
                        shared_categories=list(cats1 & cats2)
                    )
                    added += 1
        
        logger.info(f"Added {added} semantic similarity edges")
    
    def detect_communities(self, algorithm: str = 'louvain') -> Dict[int, List[str]]:
        """Detect communities in the graph.
        
        Args:
            algorithm: Community detection algorithm ('louvain', 'greedy')
        
        Returns:
            Dictionary mapping community_id -> list of articles
        """
        if self.graph.number_of_nodes() == 0:
            logger.warning("Empty graph, cannot detect communities")
            return {}
        
        if algorithm == 'louvain' and HAVE_LOUVAIN:
            # Louvain method (best modularity)
            partition = community_louvain.best_partition(self.graph, weight='weight')
        else:
            # Greedy modularity communities (NetworkX built-in)
            communities_gen = nx.community.greedy_modularity_communities(
                self.graph,
                weight='weight'
            )
            # Convert to partition dict
            partition = {}
            for comm_idx, community in enumerate(communities_gen):
                for node in community:
                    partition[node] = comm_idx
        
        # Invert to get community_id -> articles mapping
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        
        # Add community labels as node attributes
        for node, comm_id in partition.items():
            self.graph.nodes[node]['community'] = comm_id
        
        logger.info(f"Detected {len(communities)} communities")
        for comm_id, members in communities.items():
            logger.info(f"  Community {comm_id}: {len(members)} articles")
        
        return communities
    
    def get_community_label(self, community: List[str]) -> str:
        """Generate descriptive label for a community.
        
        Uses most common entity types or categories from enriched metadata.
        
        Args:
            community: List of article names in community
        
        Returns:
            Descriptive label string
        """
        if not self.enriched_metadata:
            return f"Community ({len(community)} articles)"
        
        # Collect entity types and categories
        types = []
        categories = []
        
        for article in community:
            if article in self.enriched_metadata:
                meta = self.enriched_metadata[article]
                if meta.get('entity_type_label'):
                    types.append(meta['entity_type_label'])
                categories.extend(meta.get('categories', []))
        
        # Find most common
        if types:
            from collections import Counter
            most_common_type = Counter(types).most_common(1)[0][0]
            return f"{most_common_type}s ({len(community)} articles)"
        elif categories:
            from collections import Counter
            most_common_cat = Counter(categories).most_common(1)[0][0]
            return f"{most_common_cat} ({len(community)} articles)"
        else:
            return f"Community ({len(community)} articles)"
    
    def get_central_articles(self, metric: str = 'pagerank', top_n: int = 20) -> List[Tuple[str, float]]:
        """Get most central articles using various metrics.
        
        Args:
            metric: Centrality metric ('pagerank', 'betweenness', 'closeness', 'degree')
            top_n: Number of top articles to return
        
        Returns:
            List of (article, score) tuples sorted by score
        """
        if self.graph.number_of_nodes() == 0:
            return []
        
        if metric == 'pagerank':
            centrality = nx.pagerank(self.graph, weight='weight')
        elif metric == 'betweenness':
            centrality = nx.betweenness_centrality(self.graph, weight='weight')
        elif metric == 'closeness':
            centrality = nx.closeness_centrality(self.graph)
        elif metric == 'degree':
            centrality = dict(self.graph.degree(weight='weight'))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Sort and return top N
        sorted_articles = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_articles[:top_n]
    
    def find_triangles(self) -> List[Set[str]]:
        """Find all triangles (3-cliques) in the graph.
        
        Returns:
            List of sets of 3 articles forming triangles
        """
        if self.graph.number_of_nodes() < 3:
            return []
        
        triangles = []
        for clique in nx.enumerate_all_cliques(self.graph):
            if len(clique) == 3:
                triangles.append(set(clique))
            elif len(clique) > 3:
                break  # enumerate_all_cliques returns by size
        
        logger.info(f"Found {len(triangles)} triangles in graph")
        return triangles
    
    def get_graph_summary(self) -> Dict:
        """Get summary statistics about the graph.
        
        Returns:
            Dictionary with graph metrics
        """
        if self.graph.number_of_nodes() == 0:
            return {
                'nodes': 0,
                'edges': 0,
                'density': 0.0,
                'avg_clustering': 0.0
            }
        
        # Basic stats
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        density = nx.density(self.graph)
        
        # Clustering coefficient
        try:
            avg_clustering = nx.average_clustering(self.graph, weight='weight')
        except:
            avg_clustering = 0.0
        
        # Connected components
        n_components = nx.number_connected_components(self.graph)
        
        # Largest component size
        if n_components > 0:
            largest_cc = max(nx.connected_components(self.graph), key=len)
            largest_cc_size = len(largest_cc)
        else:
            largest_cc_size = 0
        
        return {
            'nodes': n_nodes,
            'edges': n_edges,
            'density': density,
            'avg_clustering': avg_clustering,
            'n_components': n_components,
            'largest_component_size': largest_cc_size
        }
    
    def export_graphml(self, output_path: str) -> None:
        """Export graph to GraphML format for external tools (e.g., Gephi).
        
        Args:
            output_path: Path to output .graphml file
        """
        nx.write_graphml(self.graph, output_path)
        logger.info(f"Exported graph to {output_path}")
    
    def get_node_data(self, article: str) -> Dict:
        """Get comprehensive data for a single node.
        
        Args:
            article: Article name
        
        Returns:
            Dictionary with node data including metadata and graph metrics
        """
        if not self.graph.has_node(article):
            return {}
        
        data = {
            'article': article,
            'degree': self.graph.degree(article),
            'neighbors': list(self.graph.neighbors(article))
        }
        
        # Add node attributes
        data.update(self.graph.nodes[article])
        
        # Add enriched metadata if available
        if article in self.enriched_metadata:
            data['enriched'] = self.enriched_metadata[article]
        
        return data
