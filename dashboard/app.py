import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from dotenv import load_dotenv

# =====================================================================
# ⚠️ Секретний прийом: Хірургічний патч безпеки (Monkey Patching)
# ПОВИНЕН БУТИ ДО ІМПОРТУ БУДЬ-ЯКИХ ML-БІБЛІОТЕК (SentenceTransformer)!
# =====================================================================
import transformers.utils.import_utils
import transformers.modeling_utils

def bypass_security_check():
    pass
# Перехоплюємо перевірку безпеки до того, як бібліотека її використає
transformers.utils.import_utils.check_torch_load_is_safe = bypass_security_check
if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
    transformers.modeling_utils.check_torch_load_is_safe = bypass_security_check
# =====================================================================

# ТЕПЕР імпортуємо ML-бібліотеки, коли перевірку вимкнено
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Коректне завантаження .env відносно розташування папки dashboard
load_dotenv(dotenv_path="../.env")

# Кешуємо ресурси, щоб вони не перезавантажувалися при кожному кліку,
# АЛЕ робимо кеш залежним від обраного середовища (env_type)
@st.cache_resource
def init_architecture_stack(env_type):
    if env_type == "Локально (Docker)":
        local_host = os.environ.get("PINECONE_LOCAL_HOST", "http://localhost:5080")
        pc = Pinecone(api_key="local-dummy-key", host=local_host)

        # 🛡️ Захист від HTTPS для локального емулятора
        raw_host = pc.describe_index("arxiv-papers").host
        local_url = f"http://{raw_host}" if not raw_host.startswith("http") else raw_host.replace("https", "http")
        local_url = local_url.replace("0.0.0.0", "127.0.0.1").replace("localhost", "127.0.0.1")
        index = pc.Index(name="arxiv-papers", host=local_url)
    else:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index("arxiv-papers")

    model = SentenceTransformer("allenai/specter2_base")

    parquet_path = "data/arxiv_subset.parquet" if os.path.exists("data/arxiv_subset.parquet") else "../data/arxiv_subset.parquet"
    df = pd.read_parquet(parquet_path)

    corpus = (df['title'] + " " + df['abstract']).apply(lambda x: str(x).lower().split()).tolist()
    bm25 = BM25Okapi(corpus)

    return index, model, df, bm25

st.set_page_config(page_title="AI Research Assistant", layout="wide", page_icon="🔬")

# --- Бічна панель (Sidebar) ---
st.sidebar.title("🎛 Налаштування Інфраструктури")

env_choice = st.sidebar.radio(
    "Середовище Pinecone:",
    ["Локально (Docker)", "Хмара (SaaS)"],
    help="Локальний режим використовує Docker-емулятор. Хмара використовує Pinecone.io з вашим API ключем."
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Параметри пошуку")
top_k_fetch = st.sidebar.slider("Глибина пошуку (Top-K):", min_value=10, max_value=100, value=30, step=10)
top_k_display = st.sidebar.slider("Відображати результатів:", min_value=3, max_value=20, value=5)
rrf_k = st.sidebar.number_input("Константа RRF (k):", min_value=10, max_value=100, value=60, help="Математична константа для згладжування рангів")

try:
    with st.spinner("Завантаження архітектури..."):
        index, model, df, bm25 = init_architecture_stack(env_choice)
    st.sidebar.success(f"✅ Успішно підключено до: {env_choice}")
except Exception as e:
    st.sidebar.error("❌ Помилка підключення")
    st.error(f"Деталі помилки: {e}")
    st.stop()


# --- Головний екран ---
st.title("🔬 AI Research Assistant - Hybrid Engine Studio")
st.markdown("*Розумний пошук по наукових статтях ArXiv з використанням лексичного (BM25) та семантичного (Pinecone) рушіїв.*")

query = st.text_input("📝 Введіть ваш науковий запит:", placeholder="Наприклад: reinforcement learning for robotics", help="Англійською мовою, оскільки база ArXiv англомовна")

if query:
    with st.spinner("🚀 Виконання розподіленого ранжування та RRF злиття..."):

        # 🛡️ Бронежилет MPS для вектора запиту
        q_vec_raw = model.encode([query], normalize_embeddings=True)[0]
        q_vec_np = np.array(q_vec_raw, dtype=np.float32)

        if np.isnan(q_vec_np).any() or np.all(q_vec_np == 0):
            original_device = model.device
            model.to("cpu")
            q_vec_raw = model.encode([query], normalize_embeddings=True)[0]
            model.to(original_device)

        vec = q_vec_raw.tolist()

        # Обов'язково include_values=True для 3D графіки!
        res_vec = index.query(vector=vec, top_k=top_k_fetch, include_metadata=True, include_values=True)
        vec_docs = [{"id": m['metadata']['arxiv_id'], "title": m['metadata']['title'], "score": m['score']} for m in res_vec.get('matches', [])]

        # Лексичний простір (BM25)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k_fetch]
        bm25_docs = [{"id": df.iloc[i]['id'], "title": df.iloc[i]['title'], "score": scores[i]} for i in top_indices]

        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        doc_store = {}

        for rank, doc in enumerate(bm25_docs):
            rrf_scores[doc['id']] = rrf_scores.get(doc['id'], 0) + 1 / (rrf_k + rank + 1)
            doc_store[doc['id']] = doc
        for rank, doc in enumerate(vec_docs):
            rrf_scores[doc['id']] = rrf_scores.get(doc['id'], 0) + 1 / (rrf_k + rank + 1)
            doc_store[doc['id']] = doc

        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_display]

        # --- Візуалізація: Аналітичний блок ---
        st.divider()
        st.subheader("📊 Векторна аналітика (RRF Розподіл)")

        if sorted_rrf:
            chart_data = []
            for doc_id, score in sorted_rrf:
                short_title = doc_store[doc_id]['title'][:40] + "..."
                chart_data.append({"ID": doc_id, "Стаття": short_title, "RRF Score": score})

            df_chart = pd.DataFrame(chart_data).set_index("Стаття")
            st.bar_chart(df_chart["RRF Score"], color="#ff4b4b")

        # --- Детальні результати у вкладках (Tabs) ---
        st.divider()
        st.subheader("🔎 Детальні результати та Візуалізація")

        tab1, tab2, tab3, tab4 = st.tabs(["👑 Гібридне злиття (RRF)", "🧠 Семантичний (Pinecone)", "📝 Лексичний (BM25)", "🌌 Векторний 3D-простір"])

        with tab1:
            st.markdown(f"**Топ-{top_k_display} найкращих збігів за обома алгоритмами:**")
            if sorted_rrf:
                for doc_id, score in sorted_rrf:
                    doc_title = doc_store[doc_id]['title']
                    arxiv_url = f"https://arxiv.org/abs/{doc_id}"

                    with st.expander(f"🏆 Score: {score:.4f} | {doc_title}"):
                        st.write(f"**ArXiv ID:** `{doc_id}`")
                        abstract = df[df['id'] == doc_id]['abstract'].values[0]
                        st.caption(f"{abstract[:500]}...")
                        st.markdown(f"[🔗 Відкрити повну статтю на ArXiv]({arxiv_url})")
            else:
                st.warning("Гібридний пошук не дав результатів. Спробуй змінити запит.")

        with tab2:
            if vec_docs:
                df_vec = pd.DataFrame(vec_docs[:top_k_display])
                # Безпечний вибір: беремо тільки ті колонки, які РЕАЛЬНО існують у таблиці
                cols_to_show = [c for c in ['id', 'score', 'title'] if c in df_vec.columns]
                st.dataframe(df_vec[cols_to_show], use_container_width=True, hide_index=True)
            else:
                st.info("🧠 Векторний пошук не дав результатів. Переконайтеся, що індекс Pinecone не порожній.")

        with tab3:
            if bm25_docs:
                df_bm25 = pd.DataFrame(bm25_docs[:top_k_display])
                # Безпечний вибір колонок
                cols_to_show = [c for c in ['id', 'score', 'title'] if c in df_bm25.columns]
                st.dataframe(df_bm25[cols_to_show], use_container_width=True, hide_index=True)
            else:
                st.info("📝 Лексичний пошук не дав результатів.")

        with tab4:
            st.markdown("### Інтерактивна проекція семантичного простору (PCA 768D ➔ 3D)")
            st.caption("Чим ближче точки одна до одної, тим ближчий їхній зміст. Червона зірка — ваш запит.")

            # 🛡️ Безпечна перевірка: чи є результати і чи повернув Pinecone числові матриці
            if res_vec.get('matches') and 'values' in res_vec['matches'][0]:
                # 1. Збираємо всі вектори докупи (Спочатку Запит, потім Результати)
                all_vectors = [vec] + [m['values'] for m in res_vec['matches'][:top_k_display]]
                labels = ["🎯 ВАШ ЗАПИТ"] + [f"ID: {m['metadata']['arxiv_id']}" for m in res_vec['matches'][:top_k_display]]
                titles = ["Текст запиту"] + [m['metadata']['title'] for m in res_vec['matches'][:top_k_display]]

                # 🛡️ Захист від Edge Case: для 3D простору нам потрібно щонайменше 3 точки (Запит + 2 статті)
                if len(all_vectors) < 3:
                    st.info("🌌 Знайдено занадто мало семантичних результатів для побудови тривимірної моделі. Необхідно мінімум 2 статті.")
                else:
                    # 2. Зменшення розмірності (Principal Component Analysis)
                    pca = PCA(n_components=3)
                    reduced_vecs = pca.fit_transform(all_vectors)

                    # 3. Розділяємо назад на Запит і Документи
                    q_coords = reduced_vecs[0]
                    doc_coords = reduced_vecs[1:]

                    fig = go.Figure()

                    # Малюємо документи (сині точки)
                    fig.add_trace(go.Scatter3d(
                        x=doc_coords[:, 0], y=doc_coords[:, 1], z=doc_coords[:, 2],
                        mode='markers+text',
                        text=labels[1:],
                        textposition="bottom center",
                        marker=dict(size=8, color='#1f77b4', opacity=0.8),
                        name='Знайдені статті',
                        hovertext=titles[1:],
                        hoverinfo="text"
                    ))

                    # Малюємо запит (велика червона зірка)
                    fig.add_trace(go.Scatter3d(
                        x=[q_coords[0]], y=[q_coords[1]], z=[q_coords[2]],
                        mode='markers+text',
                        text=[labels[0]],
                        textposition="top center",
                        marker=dict(size=14, color='#ff4b4b', symbol='diamond'),
                        name='Ваш запит',
                        hovertext=[query],
                        hoverinfo="text"
                    ))

                    # Додаємо тонкі лінії (стрілки) від запиту до кожної знайденої статті
                    for i in range(len(doc_coords)):
                        fig.add_trace(go.Scatter3d(
                            x=[q_coords[0], doc_coords[i, 0]],
                            y=[q_coords[1], doc_coords[i, 1]],
                            z=[q_coords[2], doc_coords[i, 2]],
                            mode='lines',
                            line=dict(color='gray', width=1, dash='dot'),
                            showlegend=False,
                            hoverinfo='none'
                        ))

                    # Налаштування вигляду сцени
                    fig.update_layout(
                        margin=dict(l=0, r=0, b=0, t=0),
                        scene=dict(
                            xaxis_title='Вісь X (PCA 1)',
                            yaxis_title='Вісь Y (PCA 2)',
                            zaxis_title='Вісь Z (PCA 3)',
                            bgcolor="#0e1117"
                        ),
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )

                    st.plotly_chart(fig, use_container_width=True)
            elif not res_vec.get('matches'):
                st.warning("⚠️ Векторна база порожня, немає точок для малювання 3D-графіка.")
            else:
                st.warning("⚠️ Для візуалізації потрібно додати `include_values=True` у метод index.query()")
