# app.py — 料理本プロモーション：多重対応分析 + Claude考察文自動生成
from __future__ import annotations
import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

# 日本語フォント設定
def setup_japanese_font():
    import urllib.request
    import os
    font_path = "/tmp/NotoSansCJK.ttc"
    if not os.path.exists(font_path):
        try:
            url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTC/NotoSansCJK-Regular.ttc"
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            return
    try:
        from matplotlib import font_manager
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
    except Exception:
        pass

setup_japanese_font()

st.set_page_config(
    page_title="多重対応分析 + AI考察",
    page_icon="📚",
    layout="wide"
)

# =========================
# Claude API
# =========================
def call_claude(summary_text: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ APIキーが設定されていません。HuggingFace SecretsにANTHROPIC_API_KEYを設定してください。"

    prompt = (
        "以下は顧客データの多重対応分析結果です。\n"
        "この結果に基づいて、マーケティング観点から400〜600字の日本語考察文を書いてください。\n"
        "・どのような顧客層が共起しているか\n"
        "・プロモーション戦略への示唆\n"
        "・第1・第2次元が何を表しているか\n"
        "を含めてください。\n\n"
        + summary_text
    )

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]
    except Exception as e:
        return f"⚠️ API呼び出しエラー: {e}"

# =========================
# 多重対応分析
# =========================
def run_mca(df: pd.DataFrame, cols: list[str]):
    try:
        from prince import MCA
    except ImportError:
        st.error("prince ライブラリが必要です。requirements.txtに追加してください。")
        st.stop()

    mca = MCA(n_components=2, random_state=42)
    mca = mca.fit(df[cols])
    coords = mca.column_coordinates(df[cols])
    eigenvalues = mca.eigenvalues_
    total = sum(eigenvalues)
    explained = [v / total * 100 for v in eigenvalues]
    return mca, coords, explained

def build_summary(coords, explained, cols) -> str:
    lines = ["【多重対応分析結果サマリー】", ""]
    lines += [
        "■ 次元の説明率",
        f"  第1次元：{explained[0]:.1f}%",
        f"  第2次元：{explained[1]:.1f}%",
        f"  累積：{explained[0]+explained[1]:.1f}%", ""
    ]
    lines.append("■ カテゴリ座標（第1・第2次元）")
    for i, idx in enumerate(coords.index):
        d1 = coords.iloc[i, 0]
        d2 = coords.iloc[i, 1]
        lines.append(f"  {idx}: 第1={d1:.3f} / 第2={d2:.3f}")
    return "\n".join(lines)

def plot_mca(coords, explained, col_names) -> BytesIO:
    # 変数ごとに色を割り当て
    palette = ["#2563eb", "#16a34a", "#dc2626", "#d97706",
               "#7c3aed", "#0891b2", "#be185d"]
    col_color_map = {col: palette[i % len(palette)]
                     for i, col in enumerate(col_names)}

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, idx in enumerate(coords.index):
        x = coords.iloc[i, 0]
        y = coords.iloc[i, 1]
        # インデックス名から元の変数名を推定して色を決定
        color = "#6b7280"
        for col in col_names:
            if str(idx).startswith(col) or f"_{col}_" in str(idx):
                color = col_color_map[col]
                break

        ax.scatter(x, y, color=color, s=80, zorder=3)
        ax.annotate(str(idx), (x, y),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=color)

    ax.axhline(0, color="#cbd5e1", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#cbd5e1", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"第1次元（{explained[0]:.1f}%）", fontsize=10)
    ax.set_ylabel(f"第2次元（{explained[1]:.1f}%）", fontsize=10)
    ax.set_title("多重対応分析：カテゴリマップ", fontsize=12, pad=12)

    # 凡例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=col_color_map[col],
               markersize=8, label=col)
        for col in col_names
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# =========================
# UI
# =========================
st.title("📚 多重対応分析 + AI考察文自動生成")
st.caption("CSVをアップロードして変数を選ぶだけ。分析とAI考察文を自動で生成します。")

# --- サイドバー ---
st.sidebar.header("⚙️ 設定")

uploaded = st.sidebar.file_uploader(
    "CSVファイルをアップロード",
    type=["csv"],
    help="SJIS・UTF-8どちらも対応しています"
)

encoding = st.sidebar.selectbox(
    "文字コード",
    ["CP932（SJIS）", "UTF-8"],
    index=0
)
enc_map = {"CP932（SJIS）": "CP932", "UTF-8": "UTF-8"}

if uploaded is None:
    st.info("← 左のサイドバーからCSVをアップロードしてください。")
    st.stop()

# データ読み込み
try:
    df_raw = pd.read_csv(uploaded, encoding=enc_map[encoding])
except Exception as e:
    st.error(f"読み込みエラー: {e}")
    st.stop()

# --- 変数選択：CSVの列名を自動取得 ---
all_cols = df_raw.columns.tolist()

# 数値列を除外（多重対応分析はカテゴリ変数のみ）
cat_cols = [c for c in all_cols
            if df_raw[c].dtype == object or df_raw[c].nunique() < 20]

use_cols = st.sidebar.multiselect(
    "分析する変数を選択（2つ以上）",
    options=cat_cols,
    default=cat_cols[:3] if len(cat_cols) >= 3 else cat_cols,
    help="カテゴリ変数を2つ以上選んでください"
)

st.sidebar.markdown("---")
st.sidebar.caption(f"総列数: {len(all_cols)}列 / カテゴリ列: {len(cat_cols)}列")

# --- バリデーション ---
if len(use_cols) < 2:
    st.warning("⚠️ 分析変数を2つ以上選択してください。")
    st.stop()

df_mca = df_raw[use_cols].dropna().astype(str)

# --- データ概要 ---
st.subheader("📋 データ概要")
c1, c2, c3 = st.columns(3)
c1.metric("総サンプル数", f"{len(df_raw):,}件")
c2.metric("有効サンプル数", f"{len(df_mca):,}件")
c3.metric("選択変数", f"{len(use_cols)}個")

# 選択変数の水準数を表示
with st.expander("選択変数の詳細"):
    detail = pd.DataFrame({
        "変数名": use_cols,
        "水準数": [df_mca[c].nunique() for c in use_cols],
        "水準一覧": [" / ".join(sorted(df_mca[c].unique())[:8]) for c in use_cols]
    })
    st.dataframe(detail, use_container_width=True, hide_index=True)

st.dataframe(df_mca.head(8), use_container_width=True)

# --- 実行ボタン ---
if st.button("🔍 分析を実行する", type="primary", use_container_width=True):

    with st.spinner("多重対応分析を実行中..."):
        try:
            mca, coords, explained = run_mca(df_mca, use_cols)
        except Exception as e:
            st.error(f"分析エラー: {e}")
            st.stop()

    # グラフ
    st.subheader("📈 カテゴリマップ")
    chart_buf = plot_mca(coords, explained, use_cols)
    st.image(chart_buf, use_container_width=True)

    # 座標テーブル
    st.subheader("📊 カテゴリ座標")
    coords_display = coords.copy()
    coords_display.columns = [
        f"第1次元（{explained[0]:.1f}%）",
        f"第2次元（{explained[1]:.1f}%）"
    ]
    st.dataframe(coords_display.round(3), use_container_width=True)

    # Claude考察文
    st.subheader("💡 Claude による考察文")
    with st.spinner("Claudeが考察文を生成中..."):
        summary = build_summary(coords, explained, use_cols)
        insight = call_claude(summary)

    st.markdown(
        f"""<div style="background:#eff6ff;border-left:4px solid #2563eb;
        padding:16px 20px;border-radius:0 8px 8px 0;line-height:1.9;font-size:14px;">
        {insight.replace(chr(10), '<br>')}
        </div>""",
        unsafe_allow_html=True
    )

    # プロンプト確認用
    with st.expander("🔧 Claudeへ送ったプロンプトを見る"):
        st.code(summary, language="text")
