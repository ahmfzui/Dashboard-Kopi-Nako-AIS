"""
Dashboard Evaluasi Operasional Cabang Kopi Nako
Aspect Impact Score (AIS) · Diagnostic Heatmap · BERTopic

Cara menjalankan:
    streamlit run app.py
"""

import unicodedata
import re
import string
import os
import json

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

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
# Catatan: bobot MLR (W) TIDAK di-hardcode — dilatih ulang dari data mentah
# (lihat _hitung_w_mlr / compute_ais_pipeline) sehingga selalu konsisten
# dengan rentang tanggal yang sedang difilter.
MIN_ROWS_MLR = 50  # minimum jumlah ulasan agar regresi OLS bisa dipercaya

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
# Urutan tampilan pilihan untuk cabang yang sudah dikenal. Cabang baru (id di luar
# CABANG_MAP, mis. hasil upload) otomatis ditambahkan di akhir — lihat _cabang_urutan_dinamis().
CABANG_URUTAN_DEFAULT = ["Kemang","Cinere","Tebet","Grogol",
                          "Senayan Park","Palmerah","Abdul Muis","Ciracas"]
REQUIRED_UPLOAD_COLS = ["cabang", "tanggal", "rating"] + [
    "sent_product", "sent_price", "sent_place", "sent_promotion",
    "sent_people", "sent_process", "sent_physical_evidence",
]

def status(s):
    return "Aman" if s <= 0.050 else ("Waspada" if s <= 0.100 else "Kritis")

# Hijau → kuning/oranye → merah, konsisten dengan heatmap AIS
STATUS_WARNA  = {"Aman":"#27AE60", "Waspada":"#F2994A", "Kritis":"#C0392B"}
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

# ─── ABSA Pipeline ──────────────────────────────────────────────────────────
_BERT_NAME   = "indobenchmark/indobert-base-p1"
_ACD_PATH    = "model/best_model_s3_IndoBERT_BiLSTM_CNN.pth"
_ASC_PATH    = "model/best_model_t2_s1_NLI_Style.pth"
_MAX_LEN     = 256
_ACD_ASPECTS = ["product", "price", "place", "promotion", "people", "process", "physical_evidence"]
_ASPECT_LABEL_ABSA = {
    "product": "Product", "price": "Price", "place": "Place",
    "promotion": "Promotion", "people": "People",
    "process": "Process", "physical_evidence": "Physical Evidence",
}
_SENT_LABEL = {0: "Negatif", 1: "Netral", 2: "Positif"}
_SENT_COLOR = {"Negatif": "#E74C3C", "Netral": "#7F8C8D", "Positif": "#27AE60"}
_SENT_BG    = {"Negatif": "#FDECEA", "Netral": "#F2F3F4", "Positif": "#EAFAF1"}
_SENT_ICON  = {"Negatif": "😞", "Netral": "😐", "Positif": "😊"}


def _preprocess(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"<[^>]+>|&[a-zA-Z]+;", " ", text)
    text = re.sub(r"http\S+|www\S+|@\w+|#", " ", text)
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"([a-zA-Z])\1\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


if _TORCH_OK:
    class _ACDModel(nn.Module):
        """IndoBERT + BiLSTM + CNN (sequential) — ACD Tahap 1 Skenario 3."""
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(_BERT_NAME)
            hidden_dim, num_filters, kernel_sizes = 256, 100, [3, 4, 5]
            self.lstm = nn.LSTM(
                input_size=self.bert.config.hidden_size,
                hidden_size=hidden_dim, num_layers=1,
                batch_first=True, bidirectional=True,
            )
            self.convs   = nn.ModuleList([nn.Conv1d(hidden_dim * 2, num_filters, k) for k in kernel_sizes])
            self.norm    = nn.LayerNorm(len(kernel_sizes) * num_filters * 2)
            self.dropout = nn.Dropout(0.3)
            self.fc      = nn.Linear(len(kernel_sizes) * num_filters * 2, 7)

        def forward(self, input_ids, attention_mask):
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            lstm_out, _ = self.lstm(bert_out)
            lstm_out = lstm_out * attention_mask.unsqueeze(-1).expand(lstm_out.size()).float()
            x = lstm_out.permute(0, 2, 1)
            conved    = [F.gelu(conv(x)) for conv in self.convs]
            max_pool  = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]
            avg_pool  = [F.avg_pool1d(c, c.shape[2]).squeeze(2) for c in conved]
            x = torch.cat([torch.cat(max_pool, dim=1), torch.cat(avg_pool, dim=1)], dim=1)
            return self.fc(self.dropout(self.norm(x)))

    class _ASCModel(nn.Module):
        """IndoBERT + BiLSTM‖CNN (parallel fusion) — ASC Tahap 2 Skenario 1 NLI_Style."""
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(_BERT_NAME)
            hidden_dim, num_filters, kernel_sizes = 128, 64, [3, 4, 5]
            self.lstm  = nn.LSTM(
                input_size=self.bert.config.hidden_size,
                hidden_size=hidden_dim, num_layers=1,
                batch_first=True, bidirectional=True,
            )
            self.convs   = nn.ModuleList([nn.Conv1d(self.bert.config.hidden_size, num_filters, k) for k in kernel_sizes])
            self.dropout = nn.Dropout(0.3)
            self.fc      = nn.Linear(hidden_dim * 2 + num_filters * len(kernel_sizes), 3)

        def forward(self, input_ids, attention_mask):
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            _, (hidden, _) = self.lstm(bert_out)
            lstm_feat = torch.cat((hidden[-2], hidden[-1]), dim=1)
            conved    = [F.relu(conv(bert_out.permute(0, 2, 1))) for conv in self.convs]
            cnn_feat  = torch.cat([F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved], dim=1)
            return self.fc(self.dropout(torch.cat([lstm_feat, cnn_feat], dim=1)))

    @st.cache_resource(show_spinner=False)
    def _load_absa():
        if not (os.path.exists(_ACD_PATH) and os.path.exists(_ASC_PATH)):
            return None, None, None
        dev = torch.device("cpu")
        tok = AutoTokenizer.from_pretrained(_BERT_NAME)
        acd = _ACDModel()
        acd.load_state_dict(torch.load(_ACD_PATH, map_location=dev, weights_only=True))
        acd.eval()
        asc = _ASCModel()
        asc.load_state_dict(torch.load(_ASC_PATH, map_location=dev, weights_only=True))
        asc.eval()
        return tok, acd, asc

    def _predict_aspects(text, tok, model, threshold=0.5):
        enc = tok(_preprocess(text), truncation=True, padding="max_length",
                  max_length=_MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            probs = torch.sigmoid(model(enc["input_ids"], enc["attention_mask"])).squeeze(0).detach().numpy()
        detected = [a for a, p in zip(_ACD_ASPECTS, probs) if p >= threshold]
        return detected, dict(zip(_ACD_ASPECTS, probs.tolist()))

    def _predict_sentiment(text, aspect, tok, model):
        enc = tok(text=_preprocess(text), text_pair=aspect, truncation=True,
                  padding="max_length", max_length=_MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(enc["input_ids"], enc["attention_mask"]), dim=-1).squeeze(0).detach().numpy()
        return _SENT_LABEL[int(probs.argmax())], probs.tolist()

else:
    def _load_absa():
        return None, None, None


# ─── Parsing tanggal relatif (Google Maps) ─────────────────────────────────
# Kolom `tanggal` berisi teks relatif seperti "3 bulan lalu" / "setahun lalu"
# (bukan tanggal absolut, dan BUKAN tanggal file CSV). Untuk bisa difilter,
# teks ini diestimasi menjadi tanggal absolut relatif terhadap hari ini.
def _parse_tanggal_relatif(text, ref_date):
    text = str(text).strip().lower()
    if text.startswith("sebulan"):
        return ref_date - relativedelta(months=1)
    if text.startswith("setahun"):
        return ref_date - relativedelta(years=1)
    if text.startswith("seminggu"):
        return ref_date - relativedelta(weeks=1)
    m = re.match(r"(\d+)\s*(detik|menit|jam|hari|minggu|bulan|tahun)\s+lalu", text)
    if not m:
        return pd.NaT
    n, unit = int(m.group(1)), m.group(2)
    if unit in ("detik", "menit", "jam"):
        return ref_date
    if unit == "hari":
        return ref_date - relativedelta(days=n)
    if unit == "minggu":
        return ref_date - relativedelta(weeks=n)
    if unit == "bulan":
        return ref_date - relativedelta(months=n)
    return ref_date - relativedelta(years=n)


# ─── Load data ──────────────────────────────────────────────────────────────
_DATA_PATH_PRODUKSI = "data/dataset_mlr_ais-wow.csv"
_DATA_PATH_SIMULASI = "sample_upload/contoh_7_cabang.csv"


@st.cache_data
def load_raw_dataset(ref_date, path=_DATA_PATH_PRODUKSI):
    df = pd.read_csv(path)
    df["tanggal_estimasi"] = df["tanggal"].apply(lambda t: _parse_tanggal_relatif(t, ref_date))
    return df


def _cabang_urutan_dinamis(df_raw):
    """Urutan cabang mengikuti data yang sedang aktif (termasuk hasil upload).
    Cabang yang sudah dikenal tetap memakai urutan CABANG_URUTAN_DEFAULT;
    cabang baru (id di luar CABANG_MAP) ditambahkan di akhir."""
    id_ada = sorted(df_raw["cabang"].unique())
    dikenal = [CABANG_MAP[cid] for cid in id_ada if cid in CABANG_MAP]
    dikenal.sort(key=lambda nama: CABANG_URUTAN_DEFAULT.index(nama) if nama in CABANG_URUTAN_DEFAULT else 999)
    baru = [CABANG_MAP.get(cid, f"Cabang {cid}") for cid in id_ada if cid not in CABANG_MAP]
    return dikenal + baru


def _siapkan_data_upload(df_new, ref_date):
    """Validasi & siapkan CSV upload agar sama strukturnya dengan dataset utama.
    Return (df_siap, daftar_error)."""
    errors = []
    missing = [c for c in REQUIRED_UPLOAD_COLS if c not in df_new.columns]
    if missing:
        errors.append(f"Kolom wajib hilang: {', '.join(missing)}")
        return None, errors

    df_new = df_new.copy()
    if not pd.api.types.is_numeric_dtype(df_new["cabang"]):
        errors.append("Kolom 'cabang' harus berisi angka (id cabang).")
    if not pd.to_numeric(df_new["rating"], errors="coerce").between(1, 5).all():
        errors.append("Kolom 'rating' harus berisi angka 1–5.")
    for a in ASPEK_URUTAN:
        vals = pd.to_numeric(df_new[a], errors="coerce")
        if not vals.isin([-1, 0, 1]).all():
            errors.append(f"Kolom '{a}' harus berisi -1, 0, atau 1.")
    if errors:
        return None, errors

    if "place_name" not in df_new.columns:
        df_new["place_name"] = df_new["cabang"].map(CABANG_MAP).fillna("Cabang Baru")
    if "ulasan" not in df_new.columns:
        df_new["ulasan"] = ""
    df_new["cabang"] = df_new["cabang"].astype(int)
    df_new["tanggal_estimasi"] = df_new["tanggal"].apply(lambda t: _parse_tanggal_relatif(t, ref_date))
    return df_new, []


# ─── Penyimpanan data tambahan (mode simulasi) ─────────────────────────────
# Data yang diupload lewat mode simulasi disimpan ke disk (bukan cuma session_state)
# supaya tetap ada walau halaman di-refresh. Terpisah dari dataset produksi asli.
_DATA_TAMBAHAN_CSV = "data/uploads_tambahan.csv"
_DATA_TAMBAHAN_LOG = "data/uploads_tambahan_log.json"
_KOLOM_DATA_TAMBAHAN = ["cabang", "place_name", "tanggal", "ulasan", "rating"] + ASPEK_URUTAN


def _baca_log_upload():
    if os.path.exists(_DATA_TAMBAHAN_LOG):
        with open(_DATA_TAMBAHAN_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _tambahkan_data_tambahan(df_siap, nama_file, ukuran):
    """Simpan baris hasil upload ke _DATA_TAMBAHAN_CSV (append & tulis ulang).
    Return False kalau file yang sama persis pernah diupload sebelumnya."""
    log = _baca_log_upload()
    if any(l["nama_file"] == nama_file and l["ukuran"] == ukuran for l in log):
        return False

    df_simpan = df_siap[_KOLOM_DATA_TAMBAHAN]
    if os.path.exists(_DATA_TAMBAHAN_CSV):
        df_lama = pd.read_csv(_DATA_TAMBAHAN_CSV)
        df_gabung = pd.concat([df_lama, df_simpan], ignore_index=True)
    else:
        df_gabung = df_simpan
    df_gabung.to_csv(_DATA_TAMBAHAN_CSV, index=False)

    log.append({"nama_file": nama_file, "ukuran": ukuran, "baris": len(df_simpan)})
    with open(_DATA_TAMBAHAN_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    return True


def _hapus_data_tambahan():
    for p in (_DATA_TAMBAHAN_CSV, _DATA_TAMBAHAN_LOG):
        if os.path.exists(p):
            os.remove(p)


def _reset_simulasi_upload():
    _hapus_data_tambahan()
    st.session_state.pop("uploader_csv", None)


def load_data_tambahan(ref_date):
    """Dibaca ulang dari disk setiap rerun (tanpa cache) supaya langsung
    konsisten begitu ada upload baru atau setelah dihapus."""
    if not os.path.exists(_DATA_TAMBAHAN_CSV):
        return None
    df = pd.read_csv(_DATA_TAMBAHAN_CSV)
    if df.empty:
        return None
    df["tanggal_estimasi"] = df["tanggal"].apply(lambda t: _parse_tanggal_relatif(t, ref_date))
    return df


def _hitung_w_mlr(df_subset):
    """Melatih ulang MLR (OLS) dan menerapkan filter p-value (alpha=0.05)
    sesuai metodologi skripsi — lihat tahap3-ais-deployment.ipynb."""
    X = df_subset[ASPEK_URUTAN]
    y = df_subset["rating"]
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()
    w, pvals, coefs = {}, {}, {}
    for a in ASPEK_URUTAN:
        p = float(model.pvalues.get(a, 1.0))
        c = float(model.params.get(a, 0.0))
        pvals[a] = p
        coefs[a] = c
        w[a] = abs(c) if p < 0.05 else 0.0
    return w, pvals, coefs, float(model.rsquared)


def _hitung_ais(df_subset, w):
    total = df_subset.groupby("cabang").size()
    rows = []
    for cid, tot in total.items():
        sub = df_subset[df_subset["cabang"] == cid]
        for aspek in ASPEK_URUTAN:
            dis = sub[sub[aspek] != 0]
            afs = len(dis) / tot if tot > 0 else 0
            pos = int((dis[aspek] == 1).sum()); neg = int((dis[aspek] == -1).sum())
            ts  = pos + neg
            nss = (pos - neg) / ts if ts > 0 else 0
            wi  = w.get(aspek, 0.0)
            ais = wi * afs * (1 - nss) if wi > 0 else 0.0
            rows.append({
                "cabang": CABANG_MAP.get(cid, f"Cabang {cid}"), "aspek": ASPEK_LABEL[aspek], "aspek_key": aspek,
                "total_ulasan": int(tot), "jumlah_disebut": len(dis),
                "positif": pos, "negatif": neg,
                "AFS": round(afs, 4), "NSS": round(nss, 4), "AIS": round(ais, 4),
                "status": status(ais),
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Melatih ulang model MLR & menghitung AIS untuk rentang tanggal terpilih…")
def compute_ais_pipeline(df_raw, start_date, end_date):
    """Pipeline penuh: filter tanggal → latih ulang MLR (W) → hitung AFS/NSS/AIS.
    Di-cache per rentang tanggal supaya berpindah-pindah filter tetap responsif."""
    mask = df_raw["tanggal_estimasi"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    sub = df_raw[mask]
    n = len(sub)
    if n < MIN_ROWS_MLR or sub["rating"].nunique() < 2:
        return {"ok": False, "n": n}
    try:
        w, pvals, coefs, r2 = _hitung_w_mlr(sub)
    except Exception:
        return {"ok": False, "n": n}
    df_ais_sub = _hitung_ais(sub, w)
    return {"ok": True, "df_ais": df_ais_sub, "w": w, "pvals": pvals, "coefs": coefs, "r2": r2, "n": n}


@st.cache_data
def load_sentimen():
    return pd.read_csv("data/sentimen_per_aspek_cabang.csv")

@st.cache_data
def load_bertopic():
    r = pd.read_csv("data/ringkasan_model_bertopic_full.csv")
    k = pd.read_csv("data/hasil_ctfidf_keywords_bertopic_full.csv")
    d = pd.read_csv("data/koordinat_intertopic_distance.csv")
    return r, k, d

_ref_date = pd.Timestamp.today().normalize()
df_sent = load_sentimen()
df_ring, df_kw_all, df_ko_all = load_bertopic()

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
        f"<span style='color:{STATUS_WARNA['Aman']};font-size:18px'>●</span>&nbsp; "
        f"<b style='color:{THEME['text']}'>Aman</b>&nbsp; AIS ≤ 0.050<br>"
        f"<span style='color:{STATUS_WARNA['Waspada']};font-size:18px'>●</span>&nbsp; "
        f"<b style='color:{THEME['text']}'>Waspada</b>&nbsp; 0.051 – 0.100<br>"
        f"<span style='color:{STATUS_WARNA['Kritis']};font-size:18px'>●</span>&nbsp; "
        f"<b style='color:{THEME['text']}'>Kritis</b>&nbsp; AIS > 0.100",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    with st.expander("Mode simulasi (uji coba upload)"):
        st.caption(
            "Basis data diganti sementara ke contoh 7 cabang, tanpa mengubah data produksi asli. "
            "Data yang diupload di bawah ini disimpan ke disk khusus untuk mode simulasi, "
            "jadi tetap ada walau halaman di-refresh — sampai dihapus manual."
        )
        st.checkbox("Mulai dari dataset contoh 7 cabang", key="mode_simulasi")

        st.markdown("**Tambah data ulasan**")
        st.caption(
            "Kolom wajib: `cabang, tanggal, rating, sent_product, sent_price, sent_place, "
            "sent_promotion, sent_people, sent_process, sent_physical_evidence`. "
            "`place_name` & `ulasan` opsional."
        )
        _file = st.file_uploader("Pilih file CSV", type=["csv"], key="uploader_csv")
        if _file is not None:
            try:
                _df_baru = pd.read_csv(_file)
                _df_siap, _errs = _siapkan_data_upload(_df_baru, _ref_date)
                if _errs:
                    for e in _errs:
                        st.error(e)
                else:
                    _ditambahkan = _tambahkan_data_tambahan(_df_siap, _file.name, _file.size)
                    if _ditambahkan:
                        st.success(f"{len(_df_siap):,} baris dari '{_file.name}' disimpan & ditambahkan ke data simulasi.")
                    else:
                        st.info(f"'{_file.name}' sudah pernah diupload sebelumnya, dilewati.")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

        _log_upload = _baca_log_upload()
        if _log_upload:
            st.caption("Riwayat data tambahan (tersimpan):")
            for l in _log_upload:
                st.markdown(f"- `{l['nama_file']}` — {l['baris']:,} baris")
            st.button(
                "Hapus data tambahan (reset simulasi)",
                key="btn_reset_upload",
                use_container_width=True,
                on_click=_reset_simulasi_upload,
            )

# ─── Gabungkan data dasar + data tambahan mode simulasi ─────────────────────
_base_path = _DATA_PATH_SIMULASI if st.session_state.get("mode_simulasi") else _DATA_PATH_PRODUKSI
df_raw_base = load_raw_dataset(_ref_date, _base_path)
if st.session_state.get("mode_simulasi"):
    df_tambahan = load_data_tambahan(_ref_date)
    df_raw = pd.concat([df_raw_base, df_tambahan], ignore_index=True) if df_tambahan is not None else df_raw_base
else:
    df_raw = df_raw_base
CABANG_URUTAN = _cabang_urutan_dinamis(df_raw)

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown("## Dashboard Evaluasi Operasional Cabang Kopi Nako")
st.caption("Aspect Impact Score (AIS)  ·  Diagnostic Heatmap  ·  Distribusi Sentimen  ·  Pola Topik Pelanggan")

# ─── Filter rentang tanggal (global — menentukan kartu ringkasan & Tab 1) ───
def _pilihan_bucket_tanggal(df_raw, ref_date):
    """Opsi dropdown berbasis satuan relatif ('X bulan lalu' dst), bukan kalender
    tanggal eksak — sesuai granularitas asli data hasil scraping Google Maps."""
    tgl_min = df_raw["tanggal_estimasi"].min()
    n_tahun = max(1, int(np.ceil((ref_date - tgl_min).days / 365)) + 1)
    opts = []
    for y in range(n_tahun, 0, -1):
        opts.append((f"{y} Tahun Lalu", (ref_date - relativedelta(years=y)).date()))
    for m in range(11, 0, -1):
        opts.append((f"{m} Bulan Lalu", (ref_date - relativedelta(months=m)).date()))
    for w in range(4, 0, -1):
        opts.append((f"{w} Minggu Lalu", (ref_date - relativedelta(weeks=w)).date()))
    for d in range(6, 0, -1):
        opts.append((f"{d} Hari Lalu", (ref_date - relativedelta(days=d)).date()))
    opts.append(("Hari Ini", ref_date.date()))
    return opts

_bucket_opts   = _pilihan_bucket_tanggal(df_raw, _ref_date)
_bucket_labels = [o[0] for o in _bucket_opts]
_bucket_map    = dict(_bucket_opts)
_tgl_min = df_raw["tanggal_estimasi"].min().date()
_tgl_max = df_raw["tanggal_estimasi"].max().date()

fdate1, fdate2 = st.columns(2)
with fdate1:
    f_dari = st.selectbox(
        "Dari", _bucket_labels, index=0, key="g_dari",
        help="Rentang tanggal ulasan (estimasi, satuan relatif) — menentukan kartu ringkasan di bawah & isi Tab Overview AIS. Bukan tanggal file data.",
    )
with fdate2:
    f_sampai = st.selectbox("Sampai", _bucket_labels, index=len(_bucket_labels) - 1, key="g_sampai")

_start_d, _end_d = _bucket_map[f_dari], _bucket_map[f_sampai]
if _start_d > _end_d:
    _start_d, _end_d = _end_d, _start_d

_pipe = compute_ais_pipeline(df_raw, _start_d, _end_d)
if not _pipe["ok"]:
    st.warning(
        f"Hanya {_pipe['n']} ulasan pada rentang \"{f_dari}\" – \"{f_sampai}\" — terlalu sedikit untuk "
        f"melatih ulang model MLR secara andal (minimum {MIN_ROWS_MLR}). Menampilkan hasil dari seluruh data."
    )
    _pipe = compute_ais_pipeline(df_raw, _tgl_min, _tgl_max)

df_ais_filtered = _pipe["df_ais"]
_is_full_range = (_start_d <= _tgl_min and _end_d >= _tgl_max)
st.caption(
    f"Menampilkan **{_pipe['n']:,} ulasan** pada rentang **{f_dari} – {f_sampai}**"
    + (" (seluruh data)" if _is_full_range else "")
    + f" · Model MLR dilatih ulang dari data pada rentang ini, R² = {_pipe['r2']:.3f}."
)

# Ringkasan metrik — dinamis mengikuti filter tanggal di atas
status_max = df_ais_filtered.groupby("cabang")["AIS"].max().reset_index()
status_max["status"] = status_max["AIS"].apply(status)
n_kritis  = (status_max["status"]=="Kritis").sum()
n_waspada = (status_max["status"]=="Waspada").sum()
n_aman    = (status_max["status"]=="Aman").sum()

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Total Cabang", len(CABANG_URUTAN))
m2.metric("Total Ulasan", f"{_pipe['n']:,}")
m3.metric("Kritis",  int(n_kritis),  delta=None)
m4.metric("Waspada", int(n_waspada), delta=None)
m5.metric("Aman",    int(n_aman),    delta=None)

with st.expander("Lihat bobot (W) hasil regresi MLR untuk rentang tanggal ini"):
    _w_tbl = pd.DataFrame([
        {
            "Aspek": ASPEK_LABEL[a],
            "Koefisien Regresi": _pipe["coefs"][a],
            "p-value": _pipe["pvals"][a],
            "Signifikan (p<0.05)": "Ya" if _pipe["pvals"][a] < 0.05 else "Tidak",
            "Bobot W": _pipe["w"][a],
        }
        for a in ASPEK_URUTAN
    ]).sort_values("Bobot W", ascending=False).reset_index(drop=True)
    st.dataframe(
        _w_tbl.style.format({"Koefisien Regresi": "{:.4f}", "p-value": "{:.4f}", "Bobot W": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )

st.markdown("---")

# ─── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  Overview AIS & Heatmap  ",
    "  Distribusi Sentimen  ",
    "  Pola Topik (BERTopic)  ",
    "  Prediksi Ulasan (ABSA)  ",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW AIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.caption(
        f"Menggunakan data hasil filter tanggal di atas (**{f_dari} – {f_sampai}**, {_pipe['n']:,} ulasan). "
        "Filter Cabang & Aspek di bawah ini hanya mengubah tampilan grafik/tabel (view) dan **tidak** melatih ulang model."
    )

    # Filter row
    fc1, fc2 = st.columns(2)
    with fc1:
        f_cabang = st.selectbox("Cabang", ["Semua Cabang"] + CABANG_URUTAN, key="t1_c")
    with fc2:
        f_aspek  = st.selectbox("Aspek 7P", ["Semua Aspek"] + ASPEK_LABELS, key="t1_a")

    # ── Heatmap ─────────────────────────────────────────────────────────────
    st.markdown("### Diagnostic Heatmap")

    pivot = df_ais_filtered.pivot(index="cabang", columns="aspek", values="AIS")
    pivot = pivot.reindex(CABANG_URUTAN)[ASPEK_LABELS]

    fig_hm = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.00, "#1E8449"],   # hijau tua — sangat aman
            [0.25, "#58D68D"],   # hijau — batas Aman (AIS 0.050)
            [0.50, "#F4D03F"],   # kuning/oranye — batas Waspada (AIS 0.100)
            [0.75, "#E67E22"],
            [1.00, "#C0392B"],   # merah — kritis
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
        f"<span style='background:{STATUS_BG['Aman']};color:{STATUS_FG['Aman']};padding:4px 12px;border-radius:20px;"
        f"font-size:12px;font-weight:600;border:1px solid {THEME['badge_border']};'>Aman &nbsp; AIS ≤ 0.050</span>"
        f"<span style='background:{STATUS_BG['Waspada']};color:{STATUS_FG['Waspada']};padding:4px 12px;border-radius:20px;"
        f"font-size:12px;font-weight:600;border:1px solid {THEME['badge_border']};'>Waspada &nbsp; 0.051 – 0.100</span>"
        f"<span style='background:{STATUS_BG['Kritis']};color:{STATUS_FG['Kritis']};padding:4px 12px;border-radius:20px;"
        f"font-size:12px;font-weight:600;border:1px solid {THEME['badge_border']};'>Kritis &nbsp; AIS > 0.100</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='{soft_card_style(bg=THEME['card_soft'])}padding:14px 18px;margin-top:10px;"
        "font-size:13px;line-height:1.65;'>"
        "<b>Apa makna status ini?</b> Skor AIS = <i>W</i> (bobot pengaruh aspek terhadap rating, "
        "dari regresi MLR di atas) × <i>AFS</i> (seberapa sering aspek dibahas pelanggan) × "
        "(1 − <i>NSS</i>) (seberapa negatif sentimen pada aspek itu). Semakin besar skornya, semakin "
        "darurat kondisi aspek tersebut di cabang bersangkutan.<br><br>"
        f"<span style='color:{STATUS_FG['Aman']};font-weight:700;'>● Aman (AIS ≤ 0.050)</span> — aspek jarang "
        "dibahas dan/atau sentimennya cenderung positif, atau pengaruhnya terhadap rating tidak signifikan "
        "secara statistik (W = 0). Belum perlu tindakan segera.<br>"
        f"<span style='color:{STATUS_FG['Waspada']};font-weight:700;'>● Waspada (0.051 – 0.100)</span> — aspek "
        "mulai cukup sering dibahas dan/atau sentimennya mulai mengarah negatif. Perlu dipantau agar tidak "
        "memburuk menjadi kritis.<br>"
        f"<span style='color:{STATUS_FG['Kritis']};font-weight:700;'>● Kritis (AIS &gt; 0.100)</span> — aspek "
        "berpengaruh signifikan terhadap rating, sering dibahas pelanggan, dan sentimennya dominan negatif. "
        "Prioritas utama untuk perbaikan operasional cabang."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Bar Chart Terfilter ──────────────────────────────────────────────────
    st.markdown("### Detail Berdasarkan Filter")

    df_fil = df_ais_filtered.copy()
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
            agg = df_ais_filtered.groupby("cabang", as_index=False)["AIS"].max()
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
        df_ais_filtered.loc[df_ais_filtered.groupby("cabang")["AIS"].idxmax()]
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDIKSI ULASAN (ABSA)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Prediksi Ulasan — ABSA Pipeline")
    st.caption(
        "Masukkan satu ulasan → **ACD** mendeteksi aspek yang disinggung → "
        "**ASC** memprediksi sentimen tiap aspek."
    )

    # ── Status ketersediaan model ─────────────────────────────────────────
    acd_exists = os.path.exists(_ACD_PATH)
    asc_exists = os.path.exists(_ASC_PATH)

    if not _TORCH_OK:
        st.error(
            "Library `torch` dan `transformers` belum terinstall. "
            "Tambahkan ke `requirements.txt` lalu restart aplikasi.",
            icon="⚠️",
        )
    elif not (acd_exists and asc_exists):
        missing = []
        if not acd_exists: missing.append(f"`{_ACD_PATH}`")
        if not asc_exists: missing.append(f"`{_ASC_PATH}`")
        st.warning(
            f"File model belum ditemukan: {', '.join(missing)}. "
            "Letakkan model di folder `model/` lalu reload halaman.",
            icon="📂",
        )
        _card_style = soft_card_style(bg=THEME["card_soft"])
        st.markdown(
            f"<div style='{_card_style}padding:14px 18px;'>"
            "<b>Model yang dibutuhkan:</b><br>"
            "<code>model/best_model_s3_IndoBERT_BiLSTM_CNN.pth</code> — ACD (Aspect Category Detection)<br>"
            "<code>model/best_model_t2_s1_NLI_Style.pth</code> — ASC (Aspect Sentiment Classification)"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        # Load model sekali, di-cache oleh Streamlit
        with st.spinner("Memuat model ABSA ke memori… (pertama kali bisa 1–2 menit)"):
            _tok, _acd, _asc = _load_absa()

        if _tok is None:
            st.error("Gagal memuat model. Cek log terminal untuk detail.", icon="❌")
        else:
            # Status badge
            st.markdown(
                f"<div style='display:flex;gap:10px;margin-bottom:12px;'>"
                f"<span style='background:#EAFAF1;color:#1E8449;border:1px solid #27AE60;"
                f"border-radius:20px;padding:4px 14px;font-size:12px;font-weight:600;'>✓ ACD Model siap</span>"
                f"<span style='background:#EAFAF1;color:#1E8449;border:1px solid #27AE60;"
                f"border-radius:20px;padding:4px 14px;font-size:12px;font-weight:600;'>✓ ASC Model siap</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── Input ulasan ─────────────────────────────────────────────
            review_text = st.text_area(
                "Masukkan ulasan pelanggan",
                placeholder=(
                    "Contoh: Kopinya enak banget dan tempatnya nyaman, "
                    "tapi pelayanannya lambat dan harganya agak mahal."
                ),
                height=120,
                key="absa_review_input",
            )

            col_btn, col_thr = st.columns([2, 3])
            with col_btn:
                run_absa = st.button(
                    "Prediksi Aspek & Sentimen", type="primary",
                    use_container_width=True, key="absa_run",
                )
            with col_thr:
                acd_threshold = st.slider(
                    "Threshold deteksi aspek", 0.10, 0.90, 0.50, 0.05,
                    key="absa_threshold",
                    help="Semakin rendah → lebih banyak aspek terdeteksi (tapi lebih berisiko salah).",
                )

            if run_absa:
                if not review_text.strip():
                    st.warning("Masukkan teks ulasan terlebih dahulu.", icon="✍️")
                else:
                    st.markdown("---")

                    # ── Langkah 1: ACD ───────────────────────────────────
                    with st.spinner("Tahap 1/2 — Mendeteksi aspek (ACD)…"):
                        detected, asp_probs = _predict_aspects(
                            review_text, _tok, _acd, acd_threshold
                        )

                    # Tampilkan badge probabilitas semua aspek
                    st.markdown("#### Tahap 1 — Deteksi Aspek (ACD)")
                    badges = ""
                    for asp in _ACD_ASPECTS:
                        lbl  = _ASPECT_LABEL_ABSA[asp]
                        prob = asp_probs[asp]
                        hit  = asp in detected
                        bg   = "#EAFAF1" if hit else "#F4F6F8"
                        brd  = "#27AE60" if hit else "#BDC3C7"
                        clr  = "#1E8449" if hit else "#7F8C8D"
                        fw   = "700" if hit else "400"
                        tick = "✓ " if hit else ""
                        badges += (
                            f"<span style='background:{bg};color:{clr};border:1.5px solid {brd};"
                            f"border-radius:20px;padding:6px 14px;font-size:13px;font-weight:{fw};"
                            f"margin:4px 3px;display:inline-block;'>"
                            f"{tick}{lbl}&nbsp;<small style='opacity:.75'>({prob:.0%})</small></span>"
                        )
                    st.markdown(f"<div style='margin:8px 0 16px'>{badges}</div>", unsafe_allow_html=True)

                    if not detected:
                        st.info(
                            "Tidak ada aspek yang melampaui threshold. "
                            "Coba turunkan nilai threshold atau periksa ulasan.",
                            icon="🔍",
                        )
                    else:
                        # ── Langkah 2: ASC ───────────────────────────────
                        st.markdown(
                            f"#### Tahap 2 — Klasifikasi Sentimen (ASC) "
                            f"— {len(detected)} aspek terdeteksi"
                        )

                        with st.spinner("Tahap 2/2 — Memprediksi sentimen per aspek (ASC)…"):
                            sent_results = [
                                (_ASPECT_LABEL_ABSA[asp], *_predict_sentiment(review_text, asp, _tok, _asc))
                                for asp in detected
                            ]

                        # Grid kartu sentimen (maks 3 kolom)
                        n_cols  = min(len(sent_results), 3)
                        cols    = st.columns(n_cols)
                        _muted  = THEME["muted"]
                        for i, (asp_lbl, sent_lbl, sent_probs) in enumerate(sent_results):
                            bg_c = _SENT_BG[sent_lbl]
                            fg_c = _SENT_COLOR[sent_lbl]
                            icon = _SENT_ICON[sent_lbl]
                            with cols[i % n_cols]:
                                st.markdown(
                                    f"<div style='background:{bg_c};border:1.5px solid {fg_c};"
                                    f"border-radius:12px;padding:16px 18px;margin-bottom:10px;'>"
                                    f"<div style='font-size:12px;color:{_muted};font-weight:600;"
                                    f"text-transform:uppercase;letter-spacing:.05em;'>{asp_lbl}</div>"
                                    f"<div style='font-size:24px;font-weight:800;color:{fg_c};margin:6px 0 4px'>"
                                    f"{icon}&nbsp;{sent_lbl}</div>"
                                    f"<div style='font-size:11px;color:{_muted};'>"
                                    f"Neg {sent_probs[0]:.0%}&nbsp;·&nbsp;"
                                    f"Net {sent_probs[1]:.0%}&nbsp;·&nbsp;"
                                    f"Pos {sent_probs[2]:.0%}</div>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                        # Ringkasan teks
                        pos_asp = [r[0] for r in sent_results if r[1] == "Positif"]
                        neg_asp = [r[0] for r in sent_results if r[1] == "Negatif"]
                        net_asp = [r[0] for r in sent_results if r[1] == "Netral"]