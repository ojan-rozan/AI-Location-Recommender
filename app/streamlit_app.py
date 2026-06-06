"""
Coffee Shop Location Recommender — Streamlit UI.
"""

import os

import requests
import folium
import streamlit as st
from streamlit_folium import st_folium


api_url = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

preset = {
    "Senopati (CBD)": (-6.2297, 106.8195),
    "Tebet": (-6.2298, 106.8546),
    "Tanjung Duren": (-6.1753262,106.7891698),
    "PIK Avenue": (-6.1097, 106.7387),
}

cat_style = {
    # category: (border_color, fill_color, radius)
    "office": ("#1E40AF", "#3B82F6", 3),
    "mall": ("#6B21A8", "#A855F7", 6),
    "transit": ("#15803D", "#22C55E", 3),
    "school": ("#BE185D", "#EC4899", 3),
}


st.set_page_config(
    page_title="Coffee Shop Location AI",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    div[data-testid="stHorizontalBlock"] { align-items: flex-start; }
    iframe[title="streamlit_folium.st_folium"] { height: 550px !important; }
    div[data-testid="stElementContainer"]:has(> iframe[title="streamlit_folium.st_folium"]) {
        height: 550px !important;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# API config
def api_get(path, params=None):
    r = requests.get(f"{api_url}{path}", params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def api_post(path, body):
    r = requests.post(f"{api_url}{path}", json=body, timeout=180)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=900, show_spinner=False)
def get_map_data(lat, lng, radius_m):
    return api_get("/map-data", {"lat": lat, "lng": lng, "radius_m": radius_m})


@st.cache_data(ttl=900, show_spinner=False)
def get_features(lat, lng):
    return api_get("/features", {"lat": lat, "lng": lng})


@st.cache_data(ttl=900, show_spinner=False)
def get_nearest(lat, lng, top_n=3):
    return api_get("/nearest", {"lat": lat, "lng": lng, "top_n": top_n})

# Map builder
def build_map(lat, lng, radius_km, map_data, layer_flags):
    m = folium.Map(location=[lat, lng], zoom_start=14, tiles="cartodbpositron")

    folium.Marker(
        [lat, lng],
        tooltip="Kandidat",
        icon=folium.Icon(color="red", icon="star", prefix="fa"),
    ).add_to(m)
    folium.Circle([lat, lng], radius=500, color="red", fill=True, fill_opacity=0.05).add_to(m)
    folium.Circle([lat, lng], radius=1000, color="orange", fill=False, dash_array="5,5").add_to(m)

    if layer_flags.get("owner"):
        for r in map_data.get("owner", []):
            folium.CircleMarker(
                [r["lat"], r["lng"]],
                radius=8, color="black", fill=True,
                fill_color="#FFD700", fill_opacity=0.9, weight=2,
                popup=f"<b>{r.get('nama', '')}</b><br>{r.get('tipe', '')}",
                tooltip=r.get("nama", "Owner"),
            ).add_to(m)

    if layer_flags.get("cafe"):
        for r in map_data.get("cafes", []):
            folium.CircleMarker(
                [r["lat"], r["lng"]],
                radius=3, color="#FF8C00", fill=True,
                fill_color="#FFA500", fill_opacity=0.6, weight=1,
                popup=f"{r.get('name', '')}<br>{r.get('rating', 0)}★ ({int(r.get('reviews_count', 0) or 0)})",
            ).add_to(m)

    for cat, (border, fill, radius) in cat_style.items():
        if not layer_flags.get(cat):
            continue
        for r in map_data.get("poi", {}).get(cat, []):
            folium.CircleMarker(
                [r["lat"], r["lng"]],
                radius=radius, color=border, fill=True,
                fill_color=fill, fill_opacity=0.7, weight=1,
                popup=r.get("name", cat),
            ).add_to(m)

    return m

# Sidebar
def render_sidebar(lat, lng):
    with st.sidebar:
        st.header("Input Lokasi")
        lat = st.number_input("Latitude",  value=lat, format="%.6f", step=0.001)
        lng = st.number_input("Longitude", value=lng, format="%.6f", step=0.001)

        st.divider()
        st.subheader("Preset")
        for name, (plat, plng) in preset.items():
            if st.button(name, use_container_width=True):
                st.session_state.lat = plat
                st.session_state.lng = plng
                st.session_state.result = None
                st.rerun()

        st.divider()
        analyze = st.button("Analisis", type="primary", use_container_width=True)

    return lat, lng, analyze

# Main
def main():
    st.title("Coffee Shop Location Recommender — Jakarta")
    st.caption("AI analysis untuk kelayakan coffee shop")

    try:
        api_get("/health")
    except Exception as e:
        st.error(f"API tidak terjangkau di {api_url}. Jalankan API dulu.\n\n{e}")
        st.stop()

    if "lat" not in st.session_state:
        st.session_state.lat, st.session_state.lng = -6.2297, 106.8195
    if "result" not in st.session_state:
        st.session_state.result = None

    lat, lng, analyze_btn = render_sidebar(st.session_state.lat, st.session_state.lng)
    st.session_state.lat, st.session_state.lng = lat, lng

    col_map, col_stats = st.columns([2, 1])

    with col_map:
        st.subheader("Peta Lokasi")
        with st.expander("Layers", expanded=True):
            c1, c2, c3 = st.columns(3)
            layer_flags = {
                "owner": c1.checkbox("Owner", value=True),
                "cafe": c1.checkbox("Cafe", value=True),
                "office": c2.checkbox("Office", value=True),
                "mall": c2.checkbox("Mall", value=True),
                "transit": c3.checkbox("Transit", value=True),
                "school": c3.checkbox("School", value=False),
            }
            radius_km = st.slider("Radius (km)", 1, 5, 2)

        try:
            md = get_map_data(lat, lng, radius_km * 1000)
        except Exception as e:
            st.error(f"Gagal ambil map data: {e}")
            md = {"cafes": [], "owner": [], "poi": {}}

        m = build_map(lat, lng, radius_km, md, layer_flags)
        st_folium(m, height=550, use_container_width=True, returned_objects=[])

    with col_stats:
        st.subheader("Quick Stats")
        try:
            feats = get_features(lat, lng)
        except Exception as e:
            st.error(f"Gagal ambil features: {e}")
            feats = {}

        c1, c2 = st.columns(2)
        c1.metric("Cafe 500m", feats.get("n_competitors_500m", 0))
        c2.metric("Cafe 2km", feats.get("n_competitors_2km", 0))
        c1.metric("Office 500m",feats.get("n_offices_500m", 0))
        c2.metric("Transit 500m",feats.get("n_transit_500m", 0))
        c1.metric("Mall 2km", feats.get("n_malls_2km", 0))
        c2.metric("Owner", f"{feats.get('nearest_owner_store_m', 0):.0f}m")

        try:
            near = get_nearest(lat, lng, 3)
        except Exception as e:
            st.error(f"Gagal ambil nearest: {e}")
            near = {"competitors": [], "owner_stores": []}

        st.divider()
        st.subheader("Top 3 Kompetitor")
        for i, c in enumerate(near.get("competitors", []), 1):
            dist = c.get("distance_m", 0)
            color = "red" if dist < 200 else "orange" if dist < 500 else "gray"
            st.markdown(
                f"**{i}. {c.get('name', '-')}**  \n"
                f"⭐ {c.get('rating', 0):.1f} ({int(c.get('reviews_count', 0) or 0)}) · "
                f"<span style='color:{color}'><b>{dist:.0f}m</b></span>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.subheader("Owner Terdekat")
        for i, s in enumerate(near.get("owner_stores", []), 1):
            dist = s.get("distance_m", 0)
            if dist < 500:
                color, icon = "red", "⚠️"
            elif dist < 1000:
                color, icon = "orange", ""
            else:
                color, icon = "green", "✓"
            st.markdown(
                f"**{i}. {s.get('nama', 'Store')}** {icon}  \n"
                f"{s.get('tipe', '-')} · Rp {s.get('omzet_bulanan_juta', 0)}jt · "
                f"<span style='color:{color}'><b>{dist:.0f}m</b></span>",
                unsafe_allow_html=True,
            )

    if analyze_btn:
        with st.spinner("Predicting..."):
            try:
                st.session_state.result = api_post("/analyze", {"lat": lat, "lng": lng})
            except Exception as e:
                st.error(f"Gagal analisis: {e}")
                st.stop()

    if st.session_state.result is not None:
        st.divider()
        res = st.session_state.result

        if res.get("valid") is False:
            st.warning(res.get("message", "Lokasi di luar cakupan."))
        else:

            ml  = res["ml_data"]
            score = res["score"]

            st.subheader("Hasil Analisis")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Score", f"{score:.1f}/100")
            sc2.metric("Cafe 2km", ml.get("n_competitors_2km", 0))
            sc3.metric("Owner", f"{ml.get('nearest_owner_store_m', 0):.0f}m")

            if score >= 70:
                sc4.success("Layak")
            elif score >= 50:
                sc4.warning("Perlu validasi")
            else:
                sc4.error("Tidak layak")

            st.subheader("AI Analisis")
            if res.get("summary"):
                st.markdown(res["summary"])
            else:
                st.info("Ringkasan LLM tidak tersedia — cek GROQ_API_KEY.")


if __name__ == "__main__":
    main()