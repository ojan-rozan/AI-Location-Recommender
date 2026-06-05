"""
Coffee Shop Location Recommender — Streamlit UI.

Self-contained: logika dipanggil LANGSUNG (in-process) dari app/api.py,
TANPA butuh server API terpisah. Cocok buat Streamlit Community Cloud.

Run lokal:
    streamlit run app/streamlit_app.py

Butuh di env / Streamlit secrets:
    SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY (opsional buat AI summary)
"""

import os
import sys
import warnings
from pathlib import Path

import folium
import streamlit as st
from streamlit_folium import st_folium

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# logika inti (load model + data Supabase + predict) — dipanggil langsung
from app.api import do_features, do_nearest, do_map_data, do_analyze  # noqa: E402

st.set_page_config(
    page_title="Coffee Shop Location AI",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("☕ Coffee Shop Location Recommender — Jakarta")
    st.caption("AI analysis untuk kelayakan pembukaan coffee shop")

    if "lat" not in st.session_state:
        st.session_state.lat, st.session_state.lng = -6.2297, 106.8195
    if "result" not in st.session_state:
        st.session_state.result = None

    # ============ SIDEBAR ============
    with st.sidebar:
        st.header("📍 Input Lokasi")
        lat = st.number_input("Latitude", value=st.session_state.lat, format="%.6f", step=0.001)
        lng = st.number_input("Longitude", value=st.session_state.lng, format="%.6f", step=0.001)

        st.divider()
        st.subheader("Preset")
        presets = {
            "Senopati (CBD)": (-6.2297, 106.8195),
            "Tebet": (-6.2298, 106.8546),
            "Kemang": (-6.2603, 106.8137),
            "Kelapa Gading": (-6.1583, 106.9081),
            "PIK Avenue": (-6.1097, 106.7387),
            "Penjaringan": (-6.1255, 106.7700),
        }
        for name, (plat, plng) in presets.items():
            if st.button(name, use_container_width=True):
                st.session_state.lat, st.session_state.lng = plat, plng
                st.session_state.result = None
                st.rerun()

        st.divider()
        analyze_btn = st.button("🔍 Analisis", type="primary", use_container_width=True)

    st.session_state.lat, st.session_state.lng = lat, lng

    col1, col2 = st.columns([2, 1])

    # === MAP ===
    with col1:
        st.subheader("🗺️ Peta Lokasi")

        with st.expander("⚙️ Layers", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                show_owner = st.checkbox("🟡 Owner", value=True)
                show_cafe = st.checkbox("🟠 Cafe", value=True)
            with c2:
                show_office = st.checkbox("🔵 Office", value=True)
                show_mall = st.checkbox("🟣 Mall", value=True)
            with c3:
                show_transit = st.checkbox("🟢 Transit", value=True)
                show_school = st.checkbox("🩷 School", value=False)
            radius_km = st.slider("Radius (km)", 1, 5, 2)

        try:
            md = do_map_data(lat, lng, radius_km * 1000)
        except Exception as e:
            st.error(f"❌ Gagal load data: {e}")
            st.stop()

        m = folium.Map(location=[lat, lng], zoom_start=14, tiles="cartodbpositron")
        folium.Marker(
            [lat, lng], tooltip="📍 Kandidat",
            icon=folium.Icon(color="red", icon="star", prefix="fa"),
        ).add_to(m)
        folium.Circle([lat, lng], radius=500, color="red", fill=True, fill_opacity=0.05).add_to(m)
        folium.Circle([lat, lng], radius=1000, color="orange", fill=False, dash_array="5,5").add_to(m)

        if show_owner:
            for r in md["owner"]:
                folium.CircleMarker(
                    [r["lat"], r["lng"]], radius=8,
                    color="black", fill=True, fill_color="#FFD700",
                    fill_opacity=0.9, weight=2,
                    popup=f"<b>{r.get('nama', '')}</b><br>{r.get('tipe', '')}",
                    tooltip=r.get("nama", "Owner"),
                ).add_to(m)

        if show_cafe:
            for r in md["cafes"]:
                rating = r.get("rating") or 0
                reviews = int(r.get("reviews_count") or 0)
                folium.CircleMarker(
                    [r["lat"], r["lng"]], radius=3,
                    color="#FF8C00", fill=True, fill_color="#FFA500",
                    fill_opacity=0.6, weight=1,
                    popup=f"{r.get('name', '')}<br>{rating}★ ({reviews})",
                ).add_to(m)

        poi_styles = {
            "office": ("#1E40AF", "#3B82F6", show_office),
            "mall": ("#6B21A8", "#A855F7", show_mall),
            "transit": ("#15803D", "#22C55E", show_transit),
            "school": ("#BE185D", "#EC4899", show_school),
        }
        for cat, (border, fill, show) in poi_styles.items():
            if not show:
                continue
            for r in md.get("poi", {}).get(cat, []):
                folium.CircleMarker(
                    [r["lat"], r["lng"]], radius=6 if cat == "mall" else 3,
                    color=border, fill=True, fill_color=fill,
                    fill_opacity=0.7 if cat == "mall" else 0.6, weight=1,
                    popup=f"{r.get('name', cat)}",
                ).add_to(m)

        st_folium(m, height=550, use_container_width=True, returned_objects=[])

    # === QUICK STATS + NEAREST ===
    with col2:
        st.subheader("📊 Quick Stats")
        try:
            feats = do_features(lat, lng)
        except Exception as e:
            st.error(f"Gagal hitung fitur: {e}")
            feats = {}

        c1, c2 = st.columns(2)
        c1.metric("Cafe 500m", feats.get("n_competitors_500m", 0))
        c2.metric("Cafe 2km", feats.get("n_competitors_2km", 0))
        c1.metric("Office 500m", feats.get("n_offices_500m", 0))
        c2.metric("Transit 500m", feats.get("n_transit_500m", 0))
        c1.metric("Mall 2km", feats.get("n_malls_2km", 0))
        c2.metric("Owner", f"{feats.get('nearest_owner_store_m', 0):.0f}m")

        near = do_nearest(lat, lng, top_n=3)

        st.divider()
        st.subheader("🏪 Top 3 Kompetitor")
        for i, c in enumerate(near["competitors"], 1):
            dist = c.get("distance_m", 0)
            d_color = "red" if dist < 200 else "orange" if dist < 500 else "gray"
            st.markdown(
                f"**{i}. {c.get('name', '-')}**  \n"
                f"⭐ {c.get('rating', 0):.1f} ({int(c.get('reviews_count', 0))}) · "
                f"<span style='color:{d_color}'><b>{dist:.0f}m</b></span>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.subheader("🏠 Owner Terdekat")
        for i, s in enumerate(near["owner_stores"], 1):
            dist = s.get("distance_m", 0)
            if dist < 500:
                d_color, icon = "red", "⚠️"
            elif dist < 1000:
                d_color, icon = "orange", ""
            else:
                d_color, icon = "green", "✓"
            st.markdown(
                f"**{i}. {s.get('nama', 'Store')}** {icon}  \n"
                f"{s.get('tipe', '-')} · Rp {s.get('omzet_bulanan_juta', 0)}jt · "
                f"<span style='color:{d_color}'><b>{dist:.0f}m</b></span>",
                unsafe_allow_html=True,
            )

    # === ANALISIS ===
    if analyze_btn:
        with st.spinner("Predict + retrieve + LLM..."):
            try:
                st.session_state.result = do_analyze(lat, lng)
            except Exception as e:
                st.error(f"Gagal analisis: {e}")
                st.stop()

    if st.session_state.result is not None:
        st.divider()
        res = st.session_state.result
        ml = res["ml_data"]

        st.subheader("🎯 Hasil Analisis")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Score", f"{res['score']:.1f}/100")
        sc2.metric("Cafe 2km", ml.get("n_competitors_2km", 0))
        sc3.metric("Owner", f"{ml.get('nearest_owner_store_m', 0):.0f}m")
        score = res["score"]
        if score >= 70:
            sc4.success("✅ Layak")
        elif score >= 50:
            sc4.warning("⚠️ Validasi")
        else:
            sc4.error("❌ Tidak")

        st.subheader("🤖 AI Analisis")
        if res.get("summary"):
            st.markdown(res["summary"])
        else:
            st.info("Ringkasan LLM tidak tersedia (cek GROQ_API_KEY).")


if __name__ == "__main__":
    main()
