"""Knowledge retriever"""

import os

from supabase import create_client
from dotenv import load_dotenv

load_dotenv()


class KnowledgeRetriever:
    """Pilih + fetch knowledge docs based on ML feature output."""

    DEFAULT_TOPICS = ["general_trends", "failure_patterns"]

    def __init__(self, supabase_url=None, supabase_key=None):
        url = supabase_url or os.environ.get("SUPABASE_URL")
        key = supabase_key or os.environ.get("SUPABASE_KEY")

        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL + SUPABASE_KEY required. "
                "Set di .env atau pass langsung."
            )

        self.supabase = create_client(url, key)

    # Topic selection
    def select_topics(self, ml_data):
        """Rule-based: pilih topic relevant dari ML feature output."""
        topics = list(self.DEFAULT_TOPICS)

        n_off = ml_data.get("n_offices_500m", 0)
        if n_off > 30:
            topics.append("cbd_strategy")
        elif n_off < 5:
            topics.append("residential_strategy")

        n_comp = ml_data.get("n_competitors_500m", 0)
        if n_comp > 15:
            topics.append("high_competition")
        elif n_comp < 3:
            topics.append("low_competition_opportunity")

        if ml_data.get("n_transit_500m", 0) > 3:
            topics.append("transit_advantage")
        if ml_data.get("n_malls_2km", 0) > 0:
            topics.append("mall_anchor")
        if ml_data.get("nearest_owner_store_m", 9999) < 1000:
            topics.append("cannibalization")
        if ml_data.get("avg_competitor_rating_2km", 0) > 0:
            topics.append("rating_analysis")

        return list(set(topics))

    # retrieve
    def retrieve(self, ml_data):
        """Fetch docs dari Supabase by topic."""
        topics = self.select_topics(ml_data)

        try:
            result = self.supabase.table("documents").select("*").in_(
                "metadata->>topic", topics
            ).execute()
            return result.data, topics
        except Exception as e:
            print(f"Retrieval error: {e}")
            return [], topics