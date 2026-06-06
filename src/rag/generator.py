"""LLM generation"""

import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()


PROMPT_TEMPLATE = """Lo adalah analis lokasi F&B Indonesia. Berdasarkan data berikut, buat analisis kelayakan pembukaan coffee shop.

KOORDINAT: ({lat}, {lng})

=== ANALISIS RADIUS ===
🏪 KOMPETISI
- Cafe 500m: {n_competitors_500m} (rating avg {avg_rating_500}, {total_reviews_500} review)
- Cafe 2km: {n_competitors_2km} (rating avg {avg_rating_2km}, {total_reviews_2km} review)
- Market leader 2km: {max_reviews_2km} review

🏠 OWNER
- Store terdekat: {nearest_owner_store_m} m — {cannib_note}

🏢 DEMAND SIGNAL
- Kantor 500m: {n_offices_500m} ({office_signal})
- Mall 2km: {n_malls_2km}
- Transit 500m: {n_transit_500m} ({transit_signal})
- Sekolah 1km: {n_schools_1km}

=== ML MODEL ===
- Score: {ml_score}/100
- Top faktor: {shap_top3}

=== KONTEKS KNOWLEDGE BASE ===
{retrieved_context}

=== TUGAS ===
Tulis dalam Bahasa Indonesia, struktur:
1. **Ringkasan Eksekutif** (2-3 kalimat)
2. **Kekuatan Lokasi** (3-5 bullet, sebutkan angka konkret)
3. **Risiko & Kelemahan** (2-4 bullet dengan angka)
4. **Rekomendasi**: GO / HOLD / NO-GO + alasan
5. **Saran Tindak Lanjut** (3-5 action konkret)
"""


class SummaryGenerator:
    """Generate AI narrative pakai Groq LLM."""

    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    DEFAULT_MAX_TOKENS = 1500
    DEFAULT_TEMPERATURE = 0.3

    def __init__(self, api_key=None, model=None):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY required. Set di .env.")

        self.client = Groq(api_key=key)
        self.model = model or self.DEFAULT_MODEL

    @staticmethod
    def _cannibalization_note(distance):
        if distance < 500:
            return f"⚠️ RISIKO KANIBALISASI ({distance:.0f}m)"
        if distance < 1000:
            return f"masih dekat ({distance:.0f}m)"
        if distance < 3000:
            return f"jarak aman ({distance:.0f}m)"
        return f"tidak masalah ({distance:.0f}m)"

    @staticmethod
    def _office_signal(n):
        if n > 30:
            return "CBD padat"
        if n > 10:
            return "perkantoran moderat"
        if n > 0:
            return "perkantoran sedikit"
        return "tidak ada perkantoran"

    @staticmethod
    def _transit_signal(n):
        if n > 5:
            return "transit sangat baik"
        if n > 0:
            return "ada transit"
        return "tidak ada transit"

    # build prompting
    def build_prompt(self, lat, lng, ml_data, retrieved_docs):
        nearest = ml_data.get("nearest_owner_store_m", 9999)
        context = (
            "\n\n".join([f"- {d['content']}" for d in retrieved_docs])
            if retrieved_docs else "(no context retrieved)"
        )

        return PROMPT_TEMPLATE.format(
            lat=lat, lng=lng,
            n_competitors_500m=ml_data.get("n_competitors_500m", 0),
            avg_rating_500=round(ml_data.get("avg_competitor_rating_500", 0), 2),
            total_reviews_500=int(ml_data.get("total_competitor_reviews_500", 0)),
            n_competitors_2km=ml_data.get("n_competitors_2km", 0),
            avg_rating_2km=round(ml_data.get("avg_competitor_rating_2km", 0), 2),
            total_reviews_2km=int(ml_data.get("total_competitor_reviews_2km", 0)),
            max_reviews_2km=int(ml_data.get("max_competitor_reviews_2km", 0)),
            nearest_owner_store_m=int(nearest),
            cannib_note=self._cannibalization_note(nearest),
            n_offices_500m=ml_data.get("n_offices_500m", 0),
            office_signal=self._office_signal(ml_data.get("n_offices_500m", 0)),
            n_malls_2km=ml_data.get("n_malls_2km", 0),
            n_transit_500m=ml_data.get("n_transit_500m", 0),
            transit_signal=self._transit_signal(ml_data.get("n_transit_500m", 0)),
            n_schools_1km=ml_data.get("n_schools_1km", 0),
            ml_score=ml_data.get("score", 0),
            shap_top3=", ".join([
                f["feature"] for f in ml_data.get("top_factors", [])
            ]),
            retrieved_context=context,
        )

    # Generate
    def generate(self, lat, lng, ml_data, retrieved_docs):
        """Generate full summary (non-streaming)."""
        prompt = self.build_prompt(lat, lng, ml_data, retrieved_docs)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.DEFAULT_MAX_TOKENS,
            temperature=self.DEFAULT_TEMPERATURE,
        )
        return response.choices[0].message.content

    def generate_stream(self, lat, lng, ml_data, retrieved_docs):
        """Generate summary streaming. Yield chunks of text."""
        prompt = self.build_prompt(lat, lng, ml_data, retrieved_docs)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.DEFAULT_MAX_TOKENS,
            temperature=self.DEFAULT_TEMPERATURE,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content