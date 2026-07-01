"""
Dashboard Evaluasi Operasional Cabang Kopi Nako
Aspect Impact Score (AIS) · Diagnostic Heatmap · BERTopic

Cara menjalankan:
    streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Konfigurasi halaman ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kopi Nako — Dashboard Evaluasi Cabang",
    page_icon="https://landing-nako.stamps.co.id/media/thumb/brand-logo/KL5qy45g4VYuA924eSkAcV_size_200_webp.webp",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME = {
    "app_bg": "#F6F8FB",
    "sidebar_bg": "#EEF2F7",
    "sidebar_border": "#D9E0EA",
    "card_bg": "#FFFFFF",
    "card_soft": "#F8FAFC",
    "text": "#18202A",
    "muted": "#637085",
    "border": "#D9E0EA",
    "accent": "#FF6B5A",
    "grid": "#E6EAF0",
    "paper_bg": "#FFFFFF",
    "plot_bg": "#FAFAFA",
    "legend_bg": "rgba(255,255,255,0.88)",
    "badge_border": "#E2E8F0",
    "table_bg": "#FFFFFF",
}

def soft_card_style(bg=None, text=None, border=None):
    return (
        f"background:{bg or THEME['card_bg']};"
        f"color:{text or THEME['text']};"
        f"border:1px solid {border or THEME['border']};"
        "border-radius:10px;"
    )

st.markdown("""
<style>
    :root {
        --app-bg: %s;
        --sidebar-bg: %s;
        --sidebar-border: %s;
        --card-bg: %s;
        --card-soft: %s;
        --text-color: %s;
        --muted-text: %s;
        --border-color: %s;
        --accent-color: %s;
        --grid-color: %s;
    }

    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--app-bg);
        color: var(--text-color);
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        border-right: 1px solid var(--sidebar-border);
    }
    [data-testid="stSidebar"] * { color: var(--text-color) !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 { color: var(--text-color) !important; }
    [data-testid="stSidebar"] hr { border-color: var(--sidebar-border) !important; }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {
        color: var(--text-color);
    }

    [data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 12px 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"],
    [data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
    }

    /* Sembunyikan tombol collapse sidebar yang mengganggu */
    [data-testid="collapsedControl"] { display: none; }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        border-bottom: 2px solid var(--border-color);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 28px;
        font-size: 14px;
        font-weight: 500;
        color: var(--muted-text);
        border-bottom: 2px solid transparent;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: var(--text-color) !important;
        border-bottom-color: var(--accent-color) !important;
        font-weight: 600;
    }
    /* Hilangkan tombol collapse */
    button[kind="header"] { display: none !important; }
</style>
""" % (
    THEME["app_bg"],
    THEME["sidebar_bg"],
    THEME["sidebar_border"],
    THEME["card_bg"],
    THEME["card_soft"],
    THEME["text"],
    THEME["muted"],
    THEME["border"],
    THEME["accent"],
    THEME["grid"],
), unsafe_allow_html=True)

# ─── Konstanta ─────────────────────────────────────────────────────────────
BOBOT_W = {
    "sent_product": 0.2772, "sent_price": 0.3512, "sent_place": 0.1957,
    "sent_promotion": 0.0, "sent_people": 0.6692, "sent_process": 0.7118,
    "sent_physical_evidence": 0.3094,
}
ASPEK_LABEL = {
    "sent_product": "Product", "sent_price": "Price", "sent_place": "Place",
    "sent_promotion": "Promotion", "sent_people": "People",
    "sent_process": "Process", "sent_physical_evidence": "Physical Evidence",
}
ASPEK_URUTAN   = ["sent_people", "sent_physical_evidence", "sent_place",
                  "sent_price", "sent_process", "sent_product", "sent_promotion"]
ASPEK_LABELS   = [ASPEK_LABEL[a] for a in ASPEK_URUTAN]

CABANG_MAP = {1:"Cinere", 2:"Kemang", 3:"Tebet", 4:"Grogol",
              5:"Senayan Park", 6:"Palmerah", 7:"Abdul Muis", 8:"Ciracas"}
CABANG_URUTAN = ["Kemang","Cinere","Tebet","Grogol",
                 "Senayan Park","Palmerah","Abdul Muis","Ciracas"]

def status(s):
    return "Aman" if s <= 0.050 else ("Waspada" if s <= 0.100 else "Kritis")

STATUS_WARNA  = {"Aman":"#F5E08C", "Waspada":"#F2994A", "Kritis":"#C0392B"}
STATUS_BG     = {"Kritis":"#FDECEA", "Waspada":"#FEF0E7", "Aman":"#EAFAF1"}
STATUS_FG     = {"Kritis":"#C0392B", "Waspada":"#D35400", "Aman":"#1E8449"}

def _chart_layout(fig, height, margin, xgrid=False, ygrid=True, legend_orientation="h"):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=margin,
        paper_bgcolor=THEME["paper_bg"],
        plot_bgcolor=THEME["plot_bg"],
        font=dict(color=THEME["text"]),
        legend=dict(
            bgcolor=THEME["legend_bg"],
            bordercolor=THEME["border"],
            borderwidth=1,
            font=dict(size=12, color=THEME["text"]),
            orientation=legend_orientation,
        ),
    )
    fig.update_xaxes(showgrid=xgrid, gridcolor=THEME["grid"], zeroline=False)
    fig.update_yaxes(showgrid=ygrid, gridcolor=THEME["grid"], zeroline=False)
    return fig

# ─── Load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_ais():
    df = pd.read_csv("data/dataset_mlr_ais-wow.csv")
    total = df.groupby("cabang").size()
    rows = []
    for cid, tot in total.items():
        sub = df[df["cabang"]==cid]
        for aspek in ASPEK_URUTAN:
            dis = sub[sub[aspek]!=0]
            afs = len(dis)/tot
            pos = (dis[aspek]==1).sum(); neg = (dis[aspek]==-1).sum()
            ts  = pos+neg
            nss = (pos-neg)/ts if ts>0 else 0
            w   = BOBOT_W[aspek]
            ais = w*afs*(1-nss) if w>0 else 0.0
            rows.append({
                "cabang": CABANG_MAP[cid], "aspek": ASPEK_LABEL[aspek], "aspek_key": aspek,
                "total_ulasan": int(tot), "jumlah_disebut": len(dis),
                "positif": int(pos), "negatif": int(neg),
                "AFS": round(afs,4), "NSS": round(nss,4), "AIS": round(ais,4),
                "status": status(ais),
            })
    return pd.DataFrame(rows)

@st.cache_data
def load_sentimen():
    return pd.read_csv("data/sentimen_per_aspek_cabang.csv")

@st.cache_data
def load_bertopic():
    r = pd.read_csv("data/ringkasan_model_bertopic_full.csv")
    k = pd.read_csv("data/hasil_ctfidf_keywords_bertopic_full.csv")
    d = pd.read_csv("data/koordinat_intertopic_distance.csv")
    return r, k, d

df_ais = load_ais()
df_sent = load_sentimen()
df_ring, df_kw_all, df_ko_all = load_bertopic()
n_ulasan = 6599  # total ulasan di dataset

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://landing-nako.stamps.co.id/media/thumb/brand-logo/KL5qy45g4VYuA924eSkAcV_size_200_webp.webp",
        width=120,
    )
    st.markdown("## Dashboard Evaluasi Cabang")
    st.caption("Berbasis Aspect Impact Score & BERTopic")
    st.markdown("---")
    st.markdown("**Panduan Status AIS**")
    st.markdown(
        f"<span style='color:#F5E08C;font-size:18px'>●</span>&nbsp; "
        f"<b style='color:{THEME['text']}'>Aman</b>&nbsp; AIS ≤ 0.050<br>"
        f"<span style='color:#F2994A;font-size:18px'>●</span>&nbsp; "
        f"<b style='color:{THEME['text']}'>Waspada</b>&nbsp; 0.051 – 0.100<br>"
        f"<span style='color:#C0392B;font-size:18px'>●</span>&nbsp; "
        f"<b style='color:{THEME['text']}'>Kritis</b>&nbsp; AIS > 0.100",
        unsafe_allow_html=True,
    )
    

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown("## Dashboard Evaluasi Operasional Cabang Kopi Nako")
st.caption("Aspect Impact Score (AIS)  ·  Diagnostic Heatmap  ·  Distribusi Sentimen  ·  Pola Topik Pelanggan")

# Ringkasan metrik
status_max = df_ais.groupby("cabang")["AIS"].max().reset_index()
status_max["status"] = status_max["AIS"].apply(status)
n_kritis  = (status_max["status"]=="Kritis").sum()
n_waspada = (status_max["status"]=="Waspada").sum()
n_aman    = (status_max["status"]=="Aman").sum()

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Total Cabang", len(CABANG_URUTAN))
m2.metric("Total Ulasan", f"{n_ulasan:,}")
m3.metric("Kritis",  int(n_kritis),  delta=None)
m4.metric("Waspada", int(n_waspada), delta=None)
m5.metric("Aman",    int(n_aman),    delta=None)

st.markdown("---")

# ─── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  Overview AIS & Heatmap  ",
    "  Distribusi Sentimen  ",
    "  Pola Topik (BERTopic)  ",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW AIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # Filter row
    fc1, fc2 = st.columns(2)
    with fc1:
        f_cabang = st.selectbox("Cabang", ["Semua Cabang"] + CABANG_URUTAN, key="t1_c")
    with fc2:
        f_aspek  = st.selectbox("Aspek 7P", ["Semua Aspek"] + ASPEK_LABELS, key="t1_a")

    # ── Heatmap ─────────────────────────────────────────────────────────────
    st.markdown("### Diagnostic Heatmap")

    pivot = df_ais.pivot(index="cabang", columns="aspek", values="AIS")
    pivot = pivot.reindex(CABANG_URUTAN)[ASPEK_LABELS]

    fig_hm = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.00, "#FFFDE8"],
            [0.25, "#F5E08C"],
            [0.50, "#F2994A"],
            [1.00, "#7B241C"],
        ],
        zmin=0, zmax=0.20,
        text=pivot.values,
        texttemplate="%{text:.3f}",
        textfont={"size": 12},
        hovertemplate="<b>%{y}</b><br>%{x}<br>AIS: %{z:.4f}<extra></extra>",
        colorbar=dict(title="AIS", thickness=14, tickformat=".2f", len=0.9),
    ))
    fig_hm.update_layout(
        height=370, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(side="bottom", tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12)),
        paper_bgcolor=THEME["paper_bg"],
        plot_bgcolor=THEME["plot_bg"],
        font=dict(color=THEME["text"]),
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.markdown(
        f"<div style='display:flex;gap:12px;flex-wrap:wrap;margin-top:2px;margin-bottom:4px;'>"
        f"<span style='background:#F5E08C;color:#7D6608;padding:4px 12px;border-radius:20px;"
        f"font-size:12px;font-weight:600;border:1px solid {THEME['badge_border']};'>Aman &nbsp; AIS ≤ 0.050</span>"
        f"<span style='background:#F2994A;color:#fff;padding:4px 12px;border-radius:20px;"
        f"font-size:12px;font-weight:600;border:1px solid {THEME['badge_border']};'>Waspada &nbsp; 0.051 – 0.100</span>"
        f"<span style='background:#C0392B;color:#fff;padding:4px 12px;border-radius:20px;"
        f"font-size:12px;font-weight:600;border:1px solid {THEME['badge_border']};'>Kritis &nbsp; AIS > 0.100</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Bar Chart Terfilter ──────────────────────────────────────────────────
    st.markdown("### Detail Berdasarkan Filter")

    df_fil = df_ais.copy()
    parts  = []
    if f_cabang != "Semua Cabang":
        df_fil = df_fil[df_fil["cabang"] == f_cabang]; parts.append(f"Cabang **{f_cabang}**")
    if f_aspek  != "Semua Aspek":
        df_fil = df_fil[df_fil["aspek"] == f_aspek];   parts.append(f"Aspek **{f_aspek}**")
    st.markdown("Menampilkan: " + (" · ".join(parts) if parts else "**seluruh cabang dan aspek**"))

    colA, colB = st.columns([5, 4])

    with colA:
        if f_cabang == "Semua Cabang" and f_aspek == "Semua Aspek":
            # FIX: pakai MAX agar status tidak menyesatkan
            agg = df_ais.groupby("cabang", as_index=False)["AIS"].max()
            agg["cabang"] = pd.Categorical(agg["cabang"], CABANG_URUTAN, ordered=True)
            agg = agg.sort_values("cabang"); agg["status"] = agg["AIS"].apply(status)
            fig_b = px.bar(agg, x="cabang", y="AIS", color="status",
                           color_discrete_map=STATUS_WARNA,
                           category_orders={"cabang": CABANG_URUTAN, "status": ["Aman","Waspada","Kritis"]},
                           labels={"cabang":"Cabang","AIS":"Skor AIS Tertinggi"},
                           title="Skor AIS Tertinggi per Cabang")

        elif f_cabang == "Semua Cabang":
            ch = df_fil.set_index("cabang").reindex(CABANG_URUTAN).reset_index()
            ch["status"] = ch["AIS"].apply(status)
            fig_b = px.bar(ch, x="cabang", y="AIS", color="status",
                           color_discrete_map=STATUS_WARNA,
                           category_orders={"cabang":CABANG_URUTAN,"status":["Aman","Waspada","Kritis"]},
                           labels={"cabang":"Cabang","AIS":"Skor AIS"},
                           title=f"AIS — Aspek {f_aspek} per Cabang")

        elif f_aspek == "Semua Aspek":
            ch = df_fil.set_index("aspek").reindex(ASPEK_LABELS).reset_index()
            ch["status"] = ch["AIS"].apply(status)
            fig_b = px.bar(ch, x="aspek", y="AIS", color="status",
                           color_discrete_map=STATUS_WARNA,
                           category_orders={"aspek":ASPEK_LABELS,"status":["Aman","Waspada","Kritis"]},
                           labels={"aspek":"Aspek 7P","AIS":"Skor AIS"},
                           title=f"Profil AIS — Cabang {f_cabang}")

        else:
            v   = df_fil["AIS"].values[0] if len(df_fil)>0 else 0
            st_v = status(v)
            fig_b = px.bar(df_fil, x="cabang", y="AIS", color="status",
                           color_discrete_map=STATUS_WARNA,
                           title=f"{f_cabang} — {f_aspek}:  AIS {v:.4f}  ({st_v})")

        # Garis batas — tebal, warna kontras, label di dalam plot
        fig_b.add_shape(type="line", x0=-0.5, x1=1.5 if "aspek" in str(fig_b.layout.xaxis.title) else len(CABANG_URUTAN)-0.5,
                        y0=0.050, y1=0.050, xref="paper", yref="y",
                        line=dict(color="#E67E22", width=2, dash="dash"))
        fig_b.add_shape(type="line", x0=-0.5, x1=1.5,
                        y0=0.100, y1=0.100, xref="paper", yref="y",
                        line=dict(color="#C0392B", width=2, dash="dash"))
        fig_b.add_annotation(x=1, xref="paper", y=0.050, yref="y",
                              text="Batas Aman / Waspada", showarrow=False,
                              font=dict(size=11, color="#E67E22"),
                              bgcolor=THEME["legend_bg"], xanchor="right", yanchor="bottom")
        fig_b.add_annotation(x=1, xref="paper", y=0.100, yref="y",
                              text="Batas Waspada / Kritis", showarrow=False,
                              font=dict(size=11, color="#C0392B"),
                              bgcolor=THEME["legend_bg"], xanchor="right", yanchor="bottom")
        fig_b.update_layout(
            height=370, margin=dict(l=10,r=10,t=44,b=10),
            showlegend=True,
            legend=dict(
                orientation="v", x=0.01, y=0.99,
                xanchor="left", yanchor="top",
                bgcolor=THEME["legend_bg"],
                bordercolor=THEME["border"], borderwidth=1,
                font=dict(size=12, color=THEME["text"]), title=None,
            ),
            paper_bgcolor=THEME["paper_bg"],
            plot_bgcolor=THEME["plot_bg"],
            font=dict(color=THEME["text"]),
            yaxis=dict(gridcolor=THEME["grid"], gridwidth=1),
        )
        st.plotly_chart(fig_b, use_container_width=True)

    with colB:
        st.markdown("**Tabel detail**")
        # Tampilkan kolom ringkas agar tidak perlu scroll horizontal
        tbl = df_fil[["cabang","aspek","AIS","status"]].sort_values("AIS",ascending=False).reset_index(drop=True)
        tbl.columns = ["Cabang","Aspek","Skor AIS","Status"]

        def _color_row(row):
            s = row["Status"]
            if s == "Kritis":   c = "#FDECEA"; tc = "#C0392B"
            elif s == "Waspada": c = "#FEF0E7"; tc = "#D35400"
            else:                c = "#EAFAF1"; tc = "#1E8449"
            return ["","","",f"background-color:{c};color:{tc};font-weight:700"]

        styled = tbl.style.apply(_color_row, axis=1).format({"Skor AIS":"{:.4f}"})
        st.dataframe(styled, use_container_width=True, height=370, hide_index=True)

    st.markdown("---")

    # ── Ringkasan Prioritas ────────────────────────────────────────────────
    st.markdown("### Ringkasan Prioritas Kinerja Cabang")

    ring = (
        df_ais.loc[df_ais.groupby("cabang")["AIS"].idxmax()]
        [["cabang","aspek","AIS","status"]]
        .rename(columns={"cabang":"Cabang","aspek":"Aspek Paling Kritis",
                         "AIS":"Skor AIS Tertinggi","status":"Status"})
    )
    ring["Cabang"] = pd.Categorical(ring["Cabang"], CABANG_URUTAN, ordered=True)
    ring = ring.sort_values("Skor AIS Tertinggi", ascending=False).reset_index(drop=True)

    def _color_status(row):
        s = row["Status"]
        c = STATUS_BG.get(s,""); tc = STATUS_FG.get(s,"")
        return ["","",f"background-color:{c};color:{tc};font-weight:700",
                f"background-color:{c};color:{tc};font-weight:700"]

    ring_styled = ring.style.apply(_color_status, axis=1)\
                            .format({"Skor AIS Tertinggi":"{:.4f}"})
    st.dataframe(ring_styled, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DISTRIBUSI SENTIMEN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Distribusi Sentimen per Aspek")

    col_s1, col_s2, col_s3 = st.columns([2,2,2])
    with col_s1:
        mode = st.radio("Mode tampilan", ["Perbandingan 2 Cabang","Satu Cabang"], horizontal=True)
    with col_s2:
        c1_sent = st.selectbox("Cabang 1", CABANG_URUTAN, index=CABANG_URUTAN.index("Cinere"), key="s1")
    with col_s3:
        c2_sent = st.selectbox("Cabang 2", CABANG_URUTAN, index=CABANG_URUTAN.index("Abdul Muis"), key="s2") \
                  if mode=="Perbandingan 2 Cabang" else None

    def buat_bar_sentimen(cabang):
        sub = df_sent[df_sent["cabang"]==cabang].copy()
        sub["aspek"] = pd.Categorical(sub["aspek"], ASPEK_LABELS, ordered=True)
        sub = sub.sort_values("aspek")
        fig = go.Figure([
            go.Bar(name="Positif", x=sub["aspek"], y=sub["positif"],
                   marker_color="#2ECC71", opacity=0.88,
                   text=sub["positif"], textposition="outside", textfont=dict(size=10)),
            go.Bar(name="Negatif", x=sub["aspek"], y=sub["negatif"],
                   marker_color="#E74C3C", opacity=0.88,
                   text=sub["negatif"], textposition="outside", textfont=dict(size=10)),
        ])
        fig.update_layout(
            title=dict(text=f"<b>{cabang}</b>", font=dict(size=16)),
            barmode="group", height=360,
            margin=dict(l=10,r=10,t=48,b=10),
            yaxis_title="Jumlah Ulasan",
            paper_bgcolor=THEME["paper_bg"],
            plot_bgcolor=THEME["plot_bg"],
            font=dict(color=THEME["text"]),
            yaxis=dict(gridcolor=THEME["grid"]),
            legend=dict(orientation="h", y=1.12, x=0, bgcolor=THEME["legend_bg"], bordercolor=THEME["border"], borderwidth=1),
            showlegend=True,
        )
        fig.update_xaxes(tickangle=-25, automargin=True)
        return fig

    if mode == "Perbandingan 2 Cabang":
        ca, cb = st.columns(2)
        with ca: st.plotly_chart(buat_bar_sentimen(c1_sent), use_container_width=True)
        with cb: st.plotly_chart(buat_bar_sentimen(c2_sent), use_container_width=True)
    else:
        st.plotly_chart(buat_bar_sentimen(c1_sent), use_container_width=True)

    st.markdown("---")

    # Tabel detail sentimen
    st.markdown("**Detail distribusi sentimen**")
    cabang_list = [c1_sent] if not c2_sent else [c1_sent, c2_sent]
    tbl_s = df_sent[df_sent["cabang"].isin(cabang_list)].copy()
    tbl_s["% Negatif"] = (tbl_s["negatif"]/tbl_s["total"].replace(0,1)*100).round(1)
    tbl_s = tbl_s.rename(columns={"cabang":"Cabang","aspek":"Aspek",
                                    "positif":"Positif","negatif":"Negatif","total":"Total"})
    tbl_s["aspek_sort"] = pd.Categorical(tbl_s["Aspek"], ASPEK_LABELS, ordered=True)
    tbl_s = tbl_s.sort_values(["Cabang","aspek_sort"]).drop(columns="aspek_sort").reset_index(drop=True)

    # % Negatif: legend badge seperti heatmap AIS
    st.markdown(
        f"<div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:2px;margin-bottom:10px;'>"
        f"<span style='background:#FDECEA;color:#C0392B;padding:4px 12px;border-radius:20px;"
        f"font-size:12px;font-weight:600;border:1px solid {THEME['badge_border']};'>merah = tekanan tinggi (&gt;40%)</span>"
        f"<span style='background:#FFF3E0;color:#BF6000;padding:4px 12px;border-radius:20px;"
        f"font-size:12px;font-weight:600;border:1px solid {THEME['badge_border']};'>oranye = perlu perhatian (&gt;20%)</span>"
        f"<span style='background:{THEME['card_bg']};color:{THEME['muted']};padding:4px 12px;border-radius:20px;"
        f"font-size:12px;font-weight:600;border:1px solid {THEME['badge_border']};'>normal = kondisi aman</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    def _color_neg_pct(val):
        if val > 40:  return "background-color:#FDECEA;color:#C0392B;font-weight:700"
        elif val > 20: return "background-color:#FFF3E0;color:#BF6000;font-weight:600"
        return ""

    tbl_styled = tbl_s.style\
        .map(_color_neg_pct, subset=["% Negatif"])\
        .format({"% Negatif":"{:.1f}%"})
    st.dataframe(tbl_styled, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BERTOPIC
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Pola Topik Pelanggan (BERTopic)")

    col_b1, col_b2 = st.columns([3,1])
    with col_b1:
        bt_cabang = st.selectbox("Cabang", CABANG_URUTAN, index=CABANG_URUTAN.index("Abdul Muis"), key="bt_c")
    with col_b2:
        bt_senti_lbl = st.radio("Jenis Ulasan", ["Positif","Negatif"], horizontal=True, key="bt_s")
    bt_senti = "POS" if bt_senti_lbl=="Positif" else "NEG"

    model_info = df_ring[(df_ring["cabang_nama"]==bt_cabang) & (df_ring["sentimen"]==bt_senti)]

    if model_info.empty:
        st.warning("Data belum tersedia untuk kombinasi ini.")
    else:
        r = model_info.iloc[0]
        mi1,mi2,mi3,mi4 = st.columns(4)
        mi1.metric("Jumlah Topik",      int(r["jumlah_topik"]))
        mi2.metric("Coherence (C_v)",   f"{r['coherence_cv']:.4f}")
        mi3.metric("Dokumen Berlabel",  int(r["jumlah_dokumen"]))
        mi4.metric("min_cluster_size",  int(r["min_cluster_size"]))

        st.markdown("---")

        df_kw  = df_kw_all[(df_kw_all["cabang_nama"]==bt_cabang) & (df_kw_all["sentimen"]==bt_senti)].copy()
        df_ko  = df_ko_all[(df_ko_all["cabang_nama"]==bt_cabang) & (df_ko_all["sentimen"]==bt_senti)].copy()

        IS_POS = (bt_senti == "POS")
        CLR_DARK   = "#1A7A4A" if IS_POS else "#8B1A1A"
        CLR_LIGHT  = "#EAFAF1" if IS_POS else "#FDECEA"
        PALETTE    = px.colors.sequential.Greens[2:] if IS_POS else px.colors.sequential.Reds[2:]

        # ── Peta Kedekatan Antar Topik ────────────────────────────────
        st.markdown("#### Peta Kedekatan Antar Topik")
        col_map, col_ring = st.columns([1,1])

        with col_map:
            fig_d = go.Figure()
            fig_d.add_shape(type="line", x0=-1.2,x1=1.2, y0=0,y1=0, xref="x",yref="y",
                            line=dict(color="#DDD",width=1))
            fig_d.add_shape(type="line", x0=0,x1=0, y0=-1.2,y1=1.2, xref="x",yref="y",
                            line=dict(color="#DDD",width=1))

            max_doc = df_ko["doc_count"].max()
            for _, row in df_ko.iterrows():
                sz   = 28 + (row["doc_count"] / max_doc) * 52
                cidx = int(row["topic_id"]) % len(PALETTE)
                fig_d.add_trace(go.Scatter(
                    x=[row["x"]], y=[row["y"]],
                    mode="markers+text",
                    marker=dict(size=sz, color=PALETTE[cidx], opacity=0.85,
                                line=dict(color=CLR_DARK, width=2)),
                    text=[str(int(row["topic_id"]))],
                    textposition="middle center",
                    textfont=dict(size=13, color="white", family="Arial Black"),
                    hovertemplate=(
                        f"<b>Topik {int(row['topic_id'])}</b><br>"
                        f"{int(row['doc_count'])} dokumen<br>"
                        f"<i>{row['top5_keywords']}</i><extra></extra>"
                    ),
                    showlegend=False,
                ))

            fig_d.update_layout(
                height=440, margin=dict(l=10,r=10,t=10,b=10),
                xaxis=dict(range=[-1.25,1.25], showgrid=False, zeroline=False,
                           title="D1", tickfont=dict(size=10)),
                yaxis=dict(range=[-1.25,1.25], showgrid=False, zeroline=False,
                           title="D2", tickfont=dict(size=10)),
                paper_bgcolor=THEME["paper_bg"],
                plot_bgcolor=THEME["plot_bg"],
                font=dict(color=THEME["text"]),
            )
            st.plotly_chart(fig_d, use_container_width=True)
            st.caption("Ukuran gelembung proporsional dengan jumlah dokumen. Hover untuk detail.")

        with col_ring:
            st.markdown("**Ringkasan Topik**")
            for _, row in df_ko.sort_values("doc_count", ascending=False).iterrows():
                tid  = int(row["topic_id"]); ndok = int(row["doc_count"])
                st.markdown(
                    f'<div style="{soft_card_style(bg=THEME["card_bg"], text=THEME["text"], border=CLR_DARK)}padding:10px 14px;margin-bottom:8px;">'
                    f'<span style="color:{CLR_DARK};font-weight:700;font-size:13px;">'
                    f'Topik {tid}&nbsp;&nbsp;|&nbsp;&nbsp;{ndok} dokumen</span><br>'
                    f'<span style="color:{THEME["muted"]};font-size:12px;">{row["top5_keywords"]}</span></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── c-TF-IDF per Topik ────────────────────────────────────────
        st.markdown("#### Kata Kunci Utama per Topik (c-TF-IDF)")

        semua_tid = sorted(df_kw["topic_id"].unique())
        dok_label = {t: int(df_ko[df_ko["topic_id"]==t]["doc_count"].values[0])
                     for t in semua_tid if len(df_ko[df_ko["topic_id"]==t])>0}

        topik_pilihan = st.multiselect(
            "Topik yang ditampilkan",
            options=semua_tid,
            default=semua_tid,
            format_func=lambda t: f"Topik {t}  ({dok_label.get(t,'?')} dok)",
            key="bt_tp",
        )

        if not topik_pilihan:
            st.info("Pilih minimal satu topik.")
        else:
            df_kw_f = df_kw[df_kw["topic_id"].isin(topik_pilihan)]
            n   = len(topik_pilihan)
            nc  = min(2, n); nr = (n + 1) // 2
            subs = [f"Topik {t} | {dok_label.get(t,'?')} dok" for t in topik_pilihan]

            fig_kw = make_subplots(rows=nr, cols=nc, subplot_titles=subs,
                                   horizontal_spacing=0.14, vertical_spacing=0.18)
            for i, tid in enumerate(topik_pilihan):
                ri  = i//nc + 1; ci = i%nc + 1
                sub = df_kw_f[df_kw_f["topic_id"]==tid].sort_values("rank").head(10)
                sc  = sub["c_tf_idf_score"].values
                norm = (sc - sc.min()) / (sc.max() - sc.min() + 1e-9)
                if IS_POS:
                    clrs = [f"rgba(26,{int(80+140*v)},{int(74+60*v)},0.85)" for v in norm]
                else:
                    clrs = [f"rgba({int(100+120*v)},26,26,0.85)" for v in norm]
                fig_kw.add_trace(go.Bar(
                    x=sc, y=sub["keyword"], orientation="h",
                    marker_color=clrs,
                    text=[f"{s:.4f}" for s in sc], textposition="outside", textfont=dict(size=9),
                    showlegend=False,
                    hovertemplate="<b>%{y}</b><br>c-TF-IDF: %{x:.4f}<extra></extra>",
                ), row=ri, col=ci)
                fig_kw.update_yaxes(autorange="reversed", row=ri, col=ci)
                fig_kw.update_xaxes(showgrid=True, gridcolor=THEME["grid"], row=ri, col=ci)

            fig_kw.update_layout(
                height=max(340, nr*310),
                margin=dict(l=10,r=30,t=44,b=10),
                paper_bgcolor=THEME["paper_bg"],
                plot_bgcolor=THEME["plot_bg"],
                font=dict(color=THEME["text"]),
            )
            st.plotly_chart(fig_kw, use_container_width=True)

        # Tabel ekspandable
        with st.expander("Lihat tabel lengkap kata kunci semua topik"):
            tbl_kw = df_kw[["topic_id","doc_count","rank","keyword","c_tf_idf_score"]]\
                     .rename(columns={"topic_id":"Topik","doc_count":"Dokumen",
                                      "rank":"Rank","keyword":"Kata Kunci",
                                      "c_tf_idf_score":"Skor c-TF-IDF"})\
                     .sort_values(["Topik","Rank"]).reset_index(drop=True)
            st.dataframe(tbl_kw, use_container_width=True,
                         column_config={"Skor c-TF-IDF": st.column_config.NumberColumn(format="%.4f")})