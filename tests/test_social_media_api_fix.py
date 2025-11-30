import pytest
import sys
from unittest.mock import patch, MagicMock

# We need to be able to import from core
sys.path.append(".")

def test_simulated_social_media_api_import_without_facebook_scraper():
    """
    Test that SimulatedSocialMediaAPI can be imported and instantiated 
    even if facebook_scraper is not installed.
    """
    # Simulate facebook_scraper missing
    with patch.dict(sys.modules, {'facebook_scraper': None}):
        try:
            from core.data_sources.social_media_api import SimulatedSocialMediaAPI
        except ImportError:
            pytest.fail("Could not import SimulatedSocialMediaAPI when facebook_scraper is missing")
        
        # Instantiate with a dummy config
        config = {'api_keys': {}}
        # Mock get_api_key to avoid environment variable checks failure
        with patch('core.data_sources.social_media_api.get_api_key', return_value='dummy_key'):
            # Also need to mock tweepy to avoid authentication attempts if keys are present
            with patch('core.data_sources.social_media_api.tweepy'):
                api = SimulatedSocialMediaAPI(config)
                assert api is not None
                
                # Test that get_facebook_posts returns empty list or handles call gracefully
                # Since get_posts is None, it should return empty list or log warning
                # For SimulatedSocialMediaAPI, it has its own implementation which doesn't rely on get_posts directly,
                # but we want to ensure the base class import didn't crash.
                posts = api.get_facebook_posts("test query")
                assert isinstance(posts, list)
                assert len(posts) > 0 # Simulated API returns simulated posts
                assert posts[0]['text'].startswith("Simulated facebook post")
