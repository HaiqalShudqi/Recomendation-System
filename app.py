"""
Aplikasi Streamlit: Association Rule & Recommendation System
Dataset : Book-Crossing (BX-Users.csv, BX-Books.csv, BX-Book-Ratings.csv)
Dibuat dari notebook RecommendationSystem_Muhammad_Haiqal_Shudqi_5003241127.ipynb
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book-Crossing Recommender", page_icon="📚", layout="wide")

NAMA_FILE = {
    "users": "BX-Users.csv",
    "books": "BX-Books.csv",
    "ratings": "BX-Book-Ratings.csv",
}


# =============================================================================
# 1. LOAD DATA
# =============================================================================
def cari_file_lokal():
    """Cari ketiga file CSV di folder kerja atau folder 'data/'."""
    for folder in ["data", "."]:
        path = {k: os.path.join(folder, v) for k, v in NAMA_FILE.items()}
        if all(os.path.exists(p) for p in path.values()):
            return path
    return None


def _baca(sumber, **kwargs):
    """sumber bisa berupa path (str) atau isi file (bytes) dari uploader."""
    if isinstance(sumber, (bytes, bytearray)):
        sumber = io.BytesIO(sumber)
    return pd.read_csv(sumber, sep=";", encoding="latin-1", **kwargs)


@st.cache_data(show_spinner="Membaca & membersihkan data (file buku cukup besar, mohon tunggu)...")
def siapkan_data(kunci, sumber_users, sumber_books, sumber_ratings):
    """Load CSV + cleaning (sesuai Bagian 2-4 notebook). `kunci` dipakai sebagai cache key."""
    users = _baca(sumber_users)
    books = _baca(
        sumber_books,
        on_bad_lines="skip",
        low_memory=False,
        usecols=["ISBN", "Book-Title", "Book-Author"],
    )
    ratings = _baca(sumber_ratings)

    # Ringkasan untuk EDA
    info = {
        "n_users": len(users),
        "n_books": len(books),
        "n_ratings": len(ratings),
        "dist_rating": ratings["Book-Rating"].value_counts().sort_index(),
    }

    # Rating eksplisit saja (1-10)
    ratings_exp = ratings[ratings["Book-Rating"] > 0]
    info["n_rating_eksplisit"] = len(ratings_exp)

    # Umur: di luar 5-100 dianggap kosong
    umur = users["Age"].where((users["Age"] >= 5) & (users["Age"] <= 100))
    info["umur"] = umur.dropna()
    info["umur_kosong"] = int(umur.isnull().sum())

    # Gabung rating + judul buku
    data = ratings_exp.merge(books[["ISBN", "Book-Title", "Book-Author"]], on="ISBN")
    data["Book-Title"] = data["Book-Title"].str.replace("&amp;", "&", regex=False)
    data = data.drop_duplicates(["User-ID", "Book-Title"]).reset_index(drop=True)

    # Statistik buku & aktivitas user
    info["top_buku"] = (
        data.groupby("Book-Title")["Book-Rating"]
        .agg(jumlah_rating="count", rata_rata="mean")
        .sort_values("jumlah_rating", ascending=False)
        .head(10)
    )
    rating_per_user = data.groupby("User-ID").size()
    info["desc_user"] = rating_per_user.describe()
    info["user_1_rating"] = int((rating_per_user == 1).sum())
    info["user_20_rating"] = int((rating_per_user >= 20).sum())
    info["n_data"] = len(data)
    info["n_user_data"] = data["User-ID"].nunique()
    info["n_buku_data"] = data["Book-Title"].nunique()
    return data, info


# =============================================================================
# 2. BAGIAN A - ASSOCIATION RULE
# =============================================================================
@st.cache_data(show_spinner="Menjalankan Apriori...")
def hitung_rules(kunci, _data, batas_suka, min_user_suka, min_buku_suka, min_support, min_conf):
    liked = _data[_data["Book-Rating"] >= batas_suka]

    # user yang menyukai minimal `min_user_suka` buku
    jml_user = liked.groupby("User-ID").size()
    liked = liked[liked["User-ID"].isin(jml_user[jml_user >= min_user_suka].index)]

    # buku yang disukai minimal `min_buku_suka` user
    jml_buku = liked.groupby("Book-Title").size()
    liked = liked[liked["Book-Title"].isin(jml_buku[jml_buku >= min_buku_suka].index)]

    if liked.empty:
        return None

    basket = pd.crosstab(liked["User-ID"], liked["Book-Title"]) > 0
    basket = basket[basket.sum(axis=1) >= 2]
    if basket.empty:
        return None

    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    if frequent.empty:
        return {"basket_shape": basket.shape, "frequent": frequent, "rules": pd.DataFrame()}
    frequent["jumlah_item"] = frequent["itemsets"].apply(len)

    try:  # mlxtend versi baru membutuhkan num_itemsets
        rules = association_rules(frequent, metric="confidence", min_threshold=min_conf,
                                  num_itemsets=len(basket))
    except TypeError:  # mlxtend versi lama
        rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)

    rules = rules[rules["lift"] > 1].copy()
    if not rules.empty:
        rules["Jika menyukai"] = rules["antecedents"].apply(lambda x: " + ".join(sorted(x)))
        rules["Maka juga menyukai"] = rules["consequents"].apply(lambda x: " + ".join(sorted(x)))
        rules = rules.sort_values("lift", ascending=False)

    return {"basket_shape": basket.shape, "frequent": frequent, "rules": rules}


# =============================================================================
# 3. BAGIAN B - RECOMMENDER SYSTEM
# =============================================================================
@st.cache_resource(show_spinner="Membangun matriks user x buku & similarity...")
def bangun_model(kunci, _data, min_rating_user, min_rating_buku):
    df = _data.copy()
    user_count = df.groupby("User-ID").size()
    df = df[df["User-ID"].isin(user_count[user_count >= min_rating_user].index)]
    buku_count = df.groupby("Book-Title").size()
    df = df[df["Book-Title"].isin(buku_count[buku_count >= min_rating_buku].index)]

    if df.empty:
        return None

    matriks = df.pivot_table(index="User-ID", columns="Book-Title", values="Book-Rating").fillna(0)

    # Popularity (weighted rating ala IMDB)
    statistik = df.groupby("Book-Title")["Book-Rating"].agg(jumlah_rating="count", rata_rata="mean")
    C = df["Book-Rating"].mean()
    m = statistik["jumlah_rating"].quantile(0.75)
    v = statistik["jumlah_rating"]
    statistik["skor"] = (v / (v + m)) * statistik["rata_rata"] + (m / (v + m)) * C
    populer = statistik.sort_values("skor", ascending=False)

    # Similarity antar buku
    sim_buku = pd.DataFrame(
        cosine_similarity(matriks.T), index=matriks.columns, columns=matriks.columns
    )

    total_sel = matriks.shape[0] * matriks.shape[1]
    terisi = (matriks > 0).sum().sum()

    return {
        "df": df,
        "matriks": matriks,
        "populer": populer,
        "C": C,
        "m": m,
        "sim_buku": sim_buku,
        "isi_persen": terisi / total_sel * 100,
    }


def rekomendasi_populer(model, top_n=10, sudah_dibaca=()):
    return model["populer"].drop(index=list(sudah_dibaca), errors="ignore").head(top_n)


def buku_mirip(model, judul, top_n=5):
    hasil = model["sim_buku"][judul].sort_values(ascending=False).drop(judul)
    return hasil.head(top_n).rename("kemiripan").to_frame()


def rekomendasi_item_based(model, vektor, top_n=10):
    """vektor: Series rating user (index = judul buku, 0 = belum dirating)."""
    sim_buku = model["sim_buku"]
    skor = sim_buku.dot(vektor) / (sim_buku.abs().sum(axis=1) + 1e-9)
    sudah_dibaca = vektor[vektor > 0].index
    skor = skor.drop(sudah_dibaca)
    return skor.sort_values(ascending=False).head(top_n).rename("skor").to_frame()


def rekomendasi_user_based(model, vektor, top_n=10, jumlah_tetangga=20, exclude_user=None):
    matriks = model["matriks"]
    sim = cosine_similarity([vektor.values], matriks.values)[0]
    sim = pd.Series(sim, index=matriks.index)
    if exclude_user is not None:
        sim = sim.drop(exclude_user)
    tetangga = sim.sort_values(ascending=False).head(jumlah_tetangga)

    skor = matriks.loc[tetangga.index].mul(tetangga, axis=0).sum(axis=0) / (tetangga.sum() + 1e-9)
    sudah_dibaca = vektor[vektor > 0].index
    skor = skor.drop(sudah_dibaca)
    return skor.sort_values(ascending=False).head(top_n).rename("skor").to_frame()


def evaluasi_hit_rate(model, n_user=300, seed=42, jumlah_tetangga=20, progress=None):
    matriks = model["matriks"]
    sim_buku = model["sim_buku"]
    urutan_populer = model["populer"].index.tolist()

    rng = np.random.RandomState(seed)
    calon_user = matriks.index[(matriks >= 8).sum(axis=1) >= 1]
    n_user = min(n_user, len(calon_user))
    user_uji = rng.choice(calon_user, size=n_user, replace=False)

    hit_pop = hit_item = hit_user = 0
    for i, u in enumerate(user_uji):
        vektor = matriks.loc[u].copy()
        buku_disukai = vektor[vektor >= 8].index
        buku_sembunyi = rng.choice(buku_disukai)
        vektor[buku_sembunyi] = 0
        sudah_dibaca = vektor[vektor > 0].index

        # Popularity
        sudah = set(sudah_dibaca)
        rek_pop = [b for b in urutan_populer if b not in sudah][:10]
        hit_pop += buku_sembunyi in rek_pop

        # Item-based
        skor_item = sim_buku.dot(vektor) / (sim_buku.abs().sum(axis=1) + 1e-9)
        rek_item = skor_item.drop(sudah_dibaca).sort_values(ascending=False).head(10).index
        hit_item += buku_sembunyi in rek_item

        # User-based
        sim = cosine_similarity([vektor.values], matriks.values)[0]
        sim = pd.Series(sim, index=matriks.index).drop(u).sort_values(ascending=False).head(jumlah_tetangga)
        skor_user = matriks.loc[sim.index].mul(sim, axis=0).sum(axis=0) / (sim.sum() + 1e-9)
        rek_user = skor_user.drop(sudah_dibaca).sort_values(ascending=False).head(10).index
        hit_user += buku_sembunyi in rek_user

        if progress is not None and (i % 5 == 0 or i == n_user - 1):
            progress.progress((i + 1) / n_user, text=f"Menguji user {i + 1} dari {n_user}")

    n = len(user_uji)
    return pd.DataFrame({
        "Model": ["Popularity", "Item-based CF", "User-based CF"],
        "Hit": [hit_pop, hit_item, hit_user],
        "Hit Rate @10": [hit_pop / n, hit_item / n, hit_user / n],
    })


# =============================================================================
# 4. SIDEBAR: SUMBER DATA
# =============================================================================
st.title("📚 Book-Crossing: Association Rule & Recommendation System")
st.caption("Tugas Data Mining · Apriori, Popularity-based, Item-based CF, dan User-based CF")

with st.sidebar:
    st.header("⚙️ Sumber Data")
    upload = st.file_uploader(
        "Upload BX-Users.csv, BX-Books.csv, dan BX-Book-Ratings.csv",
        type="csv",
        accept_multiple_files=True,
    )

sumber = None
kunci_data = None

if upload:
    peta = {}
    for f in upload:
        nama = f.name.lower()
        if "rating" in nama:
            peta["ratings"] = f
        elif "user" in nama:
            peta["users"] = f
        elif "book" in nama:
            peta["books"] = f
    if len(peta) == 3:
        sumber = {k: v.getvalue() for k, v in peta.items()}
        kunci_data = ("upload",) + tuple((k, len(v)) for k, v in sorted(sumber.items()))
    else:
        st.sidebar.warning("Upload ketiga file sekaligus (Users, Books, Book-Ratings).")
else:
    lokal = cari_file_lokal()
    if lokal:
        sumber = lokal
        kunci_data = ("lokal",) + tuple((k, os.path.getsize(v)) for k, v in sorted(lokal.items()))
        st.sidebar.success("Memakai file CSV yang ditemukan di folder aplikasi.")

if sumber is None:
    st.info(
        "Upload ketiga file dataset pada sidebar, atau letakkan "
        "`BX-Users.csv`, `BX-Books.csv`, dan `BX-Book-Ratings.csv` di folder `data/` "
        "di samping `app.py`."
    )
    st.stop()

data, info = siapkan_data(kunci_data, sumber["users"], sumber["books"], sumber["ratings"])

# Parameter model
with st.sidebar:
    st.divider()
    st.header("🔧 Parameter")
    with st.expander("Association Rule (Apriori)", expanded=False):
        batas_suka = st.slider("Rating minimal dianggap 'suka'", 1, 10, 8)
        min_user_suka = st.slider("Min. buku disukai per user", 2, 10, 2)
        min_buku_suka = st.slider("Min. user yang menyukai suatu buku", 10, 100, 40)
        min_support = st.number_input("Min. support", 0.001, 0.1, 0.005, step=0.001, format="%.3f")
        min_conf = st.slider("Min. confidence", 0.05, 1.0, 0.2, step=0.05)
    with st.expander("Recommender System", expanded=False):
        min_rating_user = st.slider("Min. rating per user", 5, 50, 20)
        min_rating_buku = st.slider("Min. rating per buku", 5, 50, 20)

model = bangun_model(kunci_data, data, min_rating_user, min_rating_buku)

# =============================================================================
# 5. TAB
# =============================================================================
tab_eda, tab_rule, tab_rek, tab_eval, tab_info = st.tabs(
    ["📊 Data & EDA", "🔗 Association Rule", "🎯 Rekomendasi", "🧪 Evaluasi", "ℹ️ Teori & Kesimpulan"]
)

# ---------------------------------------------------------------- Data & EDA
with tab_eda:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah user", f"{info['n_users']:,}")
    c2.metric("Jumlah buku", f"{info['n_books']:,}")
    c3.metric("Jumlah rating", f"{info['n_ratings']:,}")
    c4.metric("Rating eksplisit (1-10)", f"{info['n_rating_eksplisit']:,}")

    st.subheader("Distribusi rating")
    st.caption("Nilai 0 = implicit (user berinteraksi tanpa memberi rating), sehingga dibuang pada analisis.")
    st.bar_chart(info["dist_rating"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribusi umur user")
        st.caption(f"Umur di luar 5-100 dianggap kosong. Umur kosong: {info['umur_kosong']:,} dari {info['n_users']:,}")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(info["umur"], bins=20)
        ax.set_xlabel("Umur")
        ax.set_ylabel("Jumlah user")
        st.pyplot(fig)
        plt.close(fig)
    with col_b:
        st.subheader("10 buku paling banyak dirating")
        st.dataframe(info["top_buku"].round(2), width="stretch")

    st.subheader("Data setelah digabung dan dibersihkan")
    d1, d2, d3 = st.columns(3)
    d1.metric("Rating", f"{info['n_data']:,}")
    d2.metric("User", f"{info['n_user_data']:,}")
    d3.metric("Judul buku", f"{info['n_buku_data']:,}")
    st.write(
        f"User yang hanya memberi 1 rating: **{info['user_1_rating']:,}** · "
        f"User dengan ≥ 20 rating: **{info['user_20_rating']:,}**. "
        "Data sangat *sparse*, sehingga user dan buku difilter agar cukup aktif."
    )
    with st.expander("Statistik jumlah rating per user"):
        st.dataframe(info["desc_user"].rename("nilai").to_frame())
    with st.expander("Contoh data"):
        st.dataframe(data.head(20), width="stretch")

# --------------------------------------------------------- Association Rule
with tab_rule:
    st.header("Bagian A: Association Rule Mining (Apriori)")
    st.write(
        "1 user = 1 transaksi, buku yang disukai = item. "
        "Rule dipilih berdasarkan min. confidence, lalu hanya yang **lift > 1** yang dipakai."
    )

    hasil_ar = hitung_rules(kunci_data, data, batas_suka, min_user_suka, min_buku_suka, min_support, min_conf)

    if hasil_ar is None:
        st.warning("Data transaksi kosong. Longgarkan parameter pada sidebar.")
    else:
        frequent = hasil_ar["frequent"]
        rules = hasil_ar["rules"]
        n_u, n_b = hasil_ar["basket_shape"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Ukuran basket (user)", f"{n_u:,}")
        m2.metric("Buku (kolom)", f"{n_b:,}")
        m3.metric("Frequent itemset", f"{len(frequent):,}")

        with st.expander("Frequent itemset terbanyak"):
            if not frequent.empty:
                fi = frequent.copy()
                fi["itemsets"] = fi["itemsets"].apply(lambda x: " + ".join(sorted(x)))
                st.dataframe(fi.sort_values("support", ascending=False).head(20), width="stretch")
                st.write("Jumlah itemset per ukuran:")
                st.bar_chart(frequent["jumlah_item"].value_counts().sort_index())

        if rules.empty:
            st.warning("Tidak ada rule dengan lift > 1. Turunkan min. support atau min. confidence.")
        else:
            st.subheader(f"Rule yang dihasilkan: {len(rules):,}")
            kolom = ["Jika menyukai", "Maka juga menyukai", "support", "confidence", "lift"]
            st.dataframe(rules[kolom].head(50), width="stretch", hide_index=True)

            st.subheader("Support vs Confidence")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sc = ax.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis")
            fig.colorbar(sc, label="Lift")
            ax.set_xlabel("Support")
            ax.set_ylabel("Confidence")
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Rekomendasi dari rule (1 buku sebagai syarat)")
            rules_1 = rules[rules["antecedents"].apply(len) == 1]
            if rules_1.empty:
                st.info("Tidak ada rule dengan 1 buku sebagai syarat.")
            else:
                pilihan = sorted(rules_1["Jika menyukai"].unique())
                judul = st.selectbox("Jika user menyukai buku:", pilihan, key="ar_judul")
                top_n_ar = st.slider("Jumlah rekomendasi", 1, 20, 5, key="ar_topn")
                hasil = (
                    rules_1[rules_1["Jika menyukai"] == judul]
                    .sort_values("confidence", ascending=False)[["Maka juga menyukai", "support", "confidence", "lift"]]
                    .head(top_n_ar)
                )
                st.dataframe(hasil, width="stretch", hide_index=True)

            st.info(
                "**Interpretasi:** rule dengan lift tinggi umumnya berasal dari buku satu seri "
                "(misalnya Harry Potter). Support kecil wajar karena pilihan buku sangat banyak "
                "dan selera user beragam."
            )

# ------------------------------------------------------------- Rekomendasi
with tab_rek:
    st.header("Bagian B: Recommender System")
    if model is None:
        st.warning("Matriks kosong. Longgarkan parameter filter pada sidebar.")
    else:
        matriks = model["matriks"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rating", f"{len(model['df']):,}")
        k2.metric("User", f"{matriks.shape[0]:,}")
        k3.metric("Buku", f"{matriks.shape[1]:,}")
        k4.metric("Sel terisi", f"{model['isi_persen']:.2f}%")

        metode = st.radio(
            "Pilih model",
            ["Popularity-based", "Item-based CF", "User-based CF", "Buku yang mirip"],
            horizontal=True,
        )

        # --- Popularity
        if metode == "Popularity-based":
            st.write(
                "Buku diurutkan dengan **skor tertimbang** (weighted rating ala IMDB): "
                "`WR = v/(v+m)·R + m/(v+m)·C`. Rekomendasinya sama untuk semua user."
            )
            st.caption(f"C (rata-rata seluruh rating) = {model['C']:.2f} · m (persentil ke-75 jumlah rating) = {model['m']:.0f}")
            top_n = st.slider("Jumlah rekomendasi", 1, 30, 10, key="pop_n")
            st.dataframe(rekomendasi_populer(model, top_n).round(3), width="stretch")

        # --- Item-based / User-based: pilih sumber user
        elif metode in ("Item-based CF", "User-based CF"):
            if metode == "Item-based CF":
                st.write("Merekomendasikan buku yang **mirip** (cosine similarity) dengan buku yang sudah dirating user.")
            else:
                st.write("Merekomendasikan buku yang disukai **user lain dengan selera mirip**.")

            mode_user = st.radio("Untuk siapa?", ["User yang sudah ada", "Pengguna baru (pilih buku sendiri)"], horizontal=True)
            top_n = st.slider("Jumlah rekomendasi", 1, 30, 10, key="cf_n")
            tetangga = 20
            if metode == "User-based CF":
                tetangga = st.slider("Jumlah tetangga (user mirip)", 5, 100, 20)

            vektor, exclude = None, None
            if mode_user == "User yang sudah ada":
                user_id = st.selectbox("User-ID", matriks.index.tolist())
                vektor = matriks.loc[user_id]
                exclude = user_id
                favorit = vektor[vektor >= 9].sort_values(ascending=False).index.tolist()[:10]
                st.markdown("**Buku yang pernah disukai user (rating ≥ 9):**")
                if favorit:
                    st.write("\n".join(f"- {b}" for b in favorit))
                else:
                    st.caption("Tidak ada buku dengan rating ≥ 9.")
            else:
                dipilih = st.multiselect("Pilih buku yang kamu sukai", matriks.columns.tolist())
                nilai = st.slider("Rating untuk buku-buku tersebut", 1, 10, 9)
                if dipilih:
                    vektor = pd.Series(0.0, index=matriks.columns)
                    vektor[dipilih] = nilai
                else:
                    st.info("Pilih minimal satu buku untuk mendapat rekomendasi.")

            if vektor is not None:
                if metode == "Item-based CF":
                    hasil = rekomendasi_item_based(model, vektor, top_n)
                else:
                    hasil = rekomendasi_user_based(model, vektor, top_n, tetangga, exclude)
                st.subheader("Hasil rekomendasi")
                st.dataframe(hasil.round(3), width="stretch")

        # --- Buku mirip
        else:
            st.write("Cari buku yang paling mirip dengan suatu buku (item-based, cosine similarity).")
            daftar = matriks.columns.tolist()
            idx = daftar.index("The Da Vinci Code") if "The Da Vinci Code" in daftar else 0
            judul = st.selectbox("Judul buku", daftar, index=idx)
            top_n = st.slider("Jumlah buku mirip", 1, 20, 5, key="mirip_n")
            st.dataframe(buku_mirip(model, judul, top_n).round(3), width="stretch")

# ----------------------------------------------------------------- Evaluasi
with tab_eval:
    st.header("Evaluasi Sederhana (Hit Rate @10)")
    st.write(
        "Untuk tiap user uji, **satu buku yang disukai (rating ≥ 8) disembunyikan**, lalu model membuat "
        "10 rekomendasi. Jika buku tersembunyi muncul di 10 rekomendasi, dihitung **hit**. "
        "Hit rate = jumlah hit / jumlah user uji."
    )
    if model is None:
        st.warning("Matriks kosong. Longgarkan parameter filter pada sidebar.")
    else:
        e1, e2, e3 = st.columns(3)
        n_uji = e1.slider("Jumlah user uji", 50, 1000, 300, step=50)
        seed = e2.number_input("Random seed", 0, 9999, 42)
        k_eval = e3.slider("Tetangga (user-based)", 5, 100, 20, key="k_eval")

        if st.button("▶️ Jalankan evaluasi", type="primary"):
            bar = st.progress(0.0, text="Memulai evaluasi...")
            st.session_state["hasil_eval"] = evaluasi_hit_rate(model, n_uji, int(seed), k_eval, bar)
            bar.empty()

        if "hasil_eval" in st.session_state:
            hasil_eval = st.session_state["hasil_eval"]
            ca, cb = st.columns(2)
            with ca:
                tampil = hasil_eval.copy()
                tampil["Hit Rate @10"] = (tampil["Hit Rate @10"] * 100).round(2).astype(str) + "%"
                st.dataframe(tampil, width="stretch", hide_index=True)
            with cb:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.bar(hasil_eval["Model"], hasil_eval["Hit Rate @10"])
                ax.set_ylabel("Hit Rate")
                ax.set_title("Perbandingan Hit Rate @10")
                st.pyplot(fig)
                plt.close(fig)
            st.caption("Hasil bisa sedikit berubah jika sampel user (seed) atau parameter diubah.")

# ------------------------------------------------------ Teori & Kesimpulan
with tab_info:
    st.header("Teori Singkat")
    st.markdown(
        """
**Association Rule** mencari pola "jika menyukai A, maka cenderung juga menyukai B".

| Ukuran | Rumus | Arti |
|---|---|---|
| Support | transaksi berisi A dan B / total transaksi | seberapa sering kombinasi muncul |
| Confidence | support(A,B) / support(A) | seberapa sering B muncul jika A ada |
| Lift | confidence(A→B) / support(B) | lift > 1 berarti A dan B terkait positif |

**Recommender System** yang dibuat:
- **Popularity-based**: buku dengan skor tertimbang tertinggi, sama untuk semua user.
- **Item-based CF**: buku yang mirip dengan buku yang disukai user.
- **User-based CF**: buku yang disukai user lain dengan selera mirip.

Kemiripan dihitung dengan **cosine similarity**.
"""
    )
    st.header("Kesimpulan (dari notebook)")
    st.markdown(
        """
1. Apriori menghasilkan pola seperti "jika menyukai buku A maka juga menyukai buku B". Rule dengan lift tertinggi umumnya berasal dari buku satu seri. Support kecil karena data sangat *sparse*.
2. Pada evaluasi hit rate@10 (300 user uji) di notebook: Popularity ±4%, User-based CF ±13%, Item-based CF ±33%. Model personal jauh lebih baik daripada rekomendasi populer, dan **Item-based CF paling baik**.
3. **Keterbatasan:** rating 0 (implicit) dibuang, item hanya diidentifikasi dari judul, tidak memakai penulis/penerbit/umur/lokasi. Pengembangan berikutnya: content-based filtering atau matrix factorization (SVD).
"""
    )
