import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add shared library to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from wikipedia.ontology import WikidataClient, DBpediaClient

class TestOntology(unittest.TestCase):
    @patch('requests.Session.get')
    def test_wikidata_get_entity_id(self, mock_get):
        # Mock response for wbgetentities
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "entities": {
                "Q9682": {
                    "id": "Q9682",
                    "type": "item"
                }
            }
        }
        mock_get.return_value = mock_response

        client = WikidataClient()
        entity_id = client.get_entity_id("Elizabeth_II")

        self.assertEqual(entity_id, "Q9682")

    @patch('SPARQLWrapper.SPARQLWrapper.query')
    def test_dbpedia_get_categories(self, mock_query):
        # Mock SPARQL response
        mock_results = MagicMock()
        mock_results.convert.return_value = {
            "results": {
                "bindings": [
                    {"category": {"value": "http://dbpedia.org/resource/Category:British_monarchs"}},
                    {"category": {"value": "http://dbpedia.org/resource/Category:Queens_regnant"}}
                ]
            }
        }
        mock_query.return_value = mock_results

        client = DBpediaClient()
        categories = client.get_categories("Elizabeth_II")

        self.assertEqual(len(categories), 2)
        self.assertIn("British monarchs", categories)
        self.assertIn("Queens regnant", categories)

if __name__ == '__main__':
    unittest.main()
