import streamlit as st
import numpy as np
import rasterio
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
import tempfile, io

#  Конфигурация 
MODEL_DIR   = Path(__file__).parent / "results" / "models"
TARGET_NAMES = {0: "Мусор", 1: "Водоросли", 2: "Пена", 3: "Вода"}
CLASS_COLORS = {0: "#E74C3C", 1: "#2ECC71",  2: "#F1C40F", 3: "#3498DB"}
BAND_NAMES   = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]


LABEL_REMAP = {
    0: -1, 1: 0, 2: 1, 3: 1, 4: 1,
    5: -1, 6: -1, 7: 3, 8: 2, 9: 3, 10: 3, 11: 2,
}

def load_ground_truth(img_fname):
    patches_dir = Path(__file__).parent / "data" / "patches"
    stem = img_fname.replace(".tif", "").replace(".tiff", "")
    parts = stem.rsplit("_", 1)
    scene = parts[0] 
    cl_path = patches_dir / scene / f"{stem}_cl.tif"
    if not cl_path.exists():
        return None
    with rasterio.open(cl_path) as src:
        lbl = src.read(1).astype(np.int16)

    out = np.full(lbl.shape, -1, dtype=np.int16)
    for src_cls, tgt_cls in LABEL_REMAP.items():
        out[lbl == src_cls] = tgt_cls
    return out

#  Загрузка моделей (кешируется) 
@st.cache_resource
def load_models():
    lgb_model = joblib.load(MODEL_DIR / "lgbm_stage1.pkl")
    xgb_model = joblib.load(MODEL_DIR / "xgb_stage2.pkl")
    with open(MODEL_DIR / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return lgb_model, xgb_model, float(meta["debris_threshold"])

# Спектральные индексы (дублирует логику из train.ipynb)
def compute_indices(img):
    """(11, H, W) float32 → (18, H, W)"""
    eps = 1e-6
    B03, B04, B05, B06 = img[2], img[3], img[4], img[5]
    B08, B8A, B11      = img[7], img[8], img[9]
    ndvi = (B8A - B04) / (B8A + B04 + eps)
    ndwi = (B03 - B8A) / (B03 + B8A + eps)
    fai  = B8A - B04 - (B11 - B04) * (865-665) / (1610-665)
    fdi  = B08 - B06 - (B11 - B06) * (833-740) / (1610-740)
    pi   = B8A / (B8A + B04 + eps)
    ndmi = (B8A - B11) / (B8A + B11 + eps)
    re   = (B05 - B04) / (B05 + B04 + eps)
    return np.concatenate([img, np.stack([ndvi, ndwi, fai, fdi, pi, ndmi, re])], axis=0)

def add_features(X):
    """(N, 18) → (N, 27): добавляет инженерные признаки"""
    eps = 1e-6
    B02=X[:,1]; B03=X[:,2]; B05=X[:,4]
    B08=X[:,7]; B8A=X[:,8]; B11=X[:,9]; B12=X[:,10]
    NDVI=X[:,11]; FAI=X[:,13]; PI=X[:,15]
    extra = np.stack([
        (B08-B05)/(B08+B05+eps),
        B11/(B8A+eps),
        B11/(B12+eps),
        (B11-B12)/(B11+B12+eps),
        B03/(B11+eps),
        FAI**2,
        PI*NDVI,
        B02/(B11+eps),
        PI / (np.abs(FAI) + eps),   
    ], axis=1).astype(np.float32)
    return np.concatenate([X, extra], axis=1)

#  Предсказание 
def predict_patch(img_18hw, lgb_model, xgb_model, threshold, lgb_debris_min=0.25):
    """
    img_18hw: (18, H, W) float32
    Возвращает маску (H, W) int32 с классами 0-3, -1 для невалидных пикселей
    """
    H, W = img_18hw.shape[1], img_18hw.shape[2]
    X = img_18hw.reshape(18, -1).T        
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    X_ext  = add_features(X)             
    p1     = lgb_model.predict_proba(X_ext)             
    X2     = np.concatenate([X_ext, p1], axis=1)        
    p_deb  = xgb_model.predict_proba(X2)[:, 1]           

    lgb_no_debris = p1.copy()
    lgb_no_debris[:, 0] = -1
    stage1_pred = np.argmax(lgb_no_debris, axis=1)

    xgb_says = p_deb >= threshold
    lgb_says  = p1[:, 0] >= lgb_debris_min
    final = np.where(xgb_says & lgb_says, 0, stage1_pred)
    return final.reshape(H, W).astype(np.int32)

# Визуализация 
def make_rgb(img_11hw):
    rgb = np.stack([img_11hw[3], img_11hw[2], img_11hw[1]], axis=-1).astype(np.float32)
    p2, p98 = np.percentile(rgb[rgb > 0], [2, 98]) if (rgb > 0).any() else (0.0, 1.0)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    return rgb

def make_mask_rgba(mask, valid_mask=None):
    colors_rgb = {
        0: (231, 76,  60,  220),
        1: (46,  204, 113, 180),
        2: (241, 196, 15,  200),
        3: (52,  152, 219, 100),
    }
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for cls, color in colors_rgb.items():
        m = (mask == cls)
        if valid_mask is not None:
            m = m & valid_mask
        rgba[m] = color
    return rgba

def plot_results(rgb, mask, gt_mask=None):
    has_gt = gt_mask is not None
    ncols = 3 if has_gt else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[i], label=TARGET_NAMES[i]) for i in range(4)
    ]

    ax_idx = 0

    axes[ax_idx].imshow(rgb)
    axes[ax_idx].set_title("Снимок (RGB)", fontsize=12)
    axes[ax_idx].axis("off")
    ax_idx += 1

    if has_gt:
        valid = gt_mask >= 0
        axes[ax_idx].imshow(rgb)
        axes[ax_idx].imshow(make_mask_rgba(gt_mask.clip(0), valid))
        n_labeled = int(valid.sum())
        pct_labeled = 100 * n_labeled / gt_mask.size
        axes[ax_idx].set_title(
            f"Разметка (Ground Truth)\n{n_labeled:,} пикс. размечено ({pct_labeled:.1f}%)",
            fontsize=10
        )
        axes[ax_idx].axis("off")
        ax_idx += 1

    axes[ax_idx].imshow(rgb)
    axes[ax_idx].imshow(make_mask_rgba(mask))
    axes[ax_idx].set_title("Предсказание модели", fontsize=12)
    axes[ax_idx].axis("off")

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, -0.05),
    )
    plt.tight_layout()
    return fig

# Статистика
def class_stats(mask):
    total = mask.size
    stats = {}
    for c in range(4):
        n = int((mask == c).sum())
        stats[TARGET_NAMES[c]] = {"pixels": n, "pct": 100 * n / total}
    return stats

#  UI 
st.set_page_config(page_title="Детектор морского мусора", page_icon="🌊", layout="wide")

st.title("🌊 Детектор морского мусора")
st.markdown("Загрузите мультиспектральный снимок Sentinel-2 (`.tif`) — модель покажет, где мусор.")

col_info, col_upload = st.columns([2, 1])

with col_info:
    st.markdown("""
    **Как использовать:**
    1. Загрузите `.tif` файл из папки `data/patches/`
    2. Модель автоматически классифицирует каждый пиксель
    3. Результат отображается в виде цветовой карты

    **Классы:**
    - 🔴 Мусор (пластик и плавающий мусор)  
    - 🟢 Водоросли  
    - 🟡 Пена  
    - 🔵 Вода  
    """)

with col_upload:
    uploaded = st.file_uploader("Загрузить .tif патч Sentinel-2", type=["tif", "tiff"])
    threshold_override = st.slider(
        "Порог уверенности (мусор)", 
        min_value=0.1, max_value=0.95, value=0.5, step=0.05,
        help="Чем выше — тем строже детекция мусора (меньше ложных срабатываний)"
    )

if uploaded is not None:
    with st.spinner("Загружаем модели..."):
        lgb_model, xgb_model, default_thr = load_models()

    thr = threshold_override

    with st.spinner("Обрабатываем снимок..."):
        import os

        fname = uploaded.name

        #  Проверка по имени файла (до чтения) 
        if fname.endswith("_cl.tif") or fname.endswith("_conf.tif"):
            st.error(
                f"❌ Загружен файл **`{fname}`** — это файл меток или уверенности.\n\n"
                f"Нужен основной файл снимка — без суффиксов `_cl` и `_conf`.\n\n"
                f"Например: `S2_1-12-19_48MYU_0.tif`"
            )
            st.stop()

        # Читаем через temp-файл (надёжнее MemoryFile на Windows)
        file_bytes = uploaded.read()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            with rasterio.open(tmp_path) as src:
                img = src.read().astype(np.float32)
                n_bands = img.shape[0]
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        st.caption(f"Файл: `{fname}` | каналов: {n_bands} | размер: {img.shape[1]}×{img.shape[2]} пикс.")

        #  Нормализуем до 11 каналов ────────────────────────────────────
        if n_bands < 11:
            pad = np.zeros((11 - n_bands, *img.shape[1:]), dtype=np.float32)
            img = np.concatenate([img, pad], axis=0)
        elif n_bands > 11:
            img = img[:11]

        img_18 = compute_indices(img)
        mask   = predict_patch(img_18, lgb_model, xgb_model, thr)
        rgb    = make_rgb(img)
        gt_mask = load_ground_truth(fname)
        stats   = class_stats(mask) 

    #  Метрики: GT vs Предсказание ──────────────────────────────────────────
    has_gt = gt_mask is not None
    valid  = gt_mask >= 0 if has_gt else None

    ICONS = {0: "🔴", 1: "🟢", 2: "🟡", 3: "🔵"}

    cols = st.columns(4)
    for c in range(4):
        name     = TARGET_NAMES[c]
        pred_n   = int((mask == c).sum())
        pred_pct = 100 * pred_n / mask.size

        with cols[c]:
            st.markdown(f"**{ICONS[c]} {name}**")
            if has_gt and valid.sum() > 0:
                gt_n   = int((gt_mask[valid] == c).sum())
                gt_pct = 100 * gt_n / valid.sum()
                delta  = pred_pct - gt_pct
                sign   = "+" if delta >= 0 else ""
                st.metric(
                    label="В датасете",
                    value=f"{gt_pct:.1f}%  ({gt_n:,} пикс.)",
                )
                st.metric(
                    label="Найдено моделью",
                    value=f"{pred_pct:.1f}%  ({pred_n:,} пикс.)",
                    delta=f"{sign}{delta:.1f}%",
                    delta_color="inverse" if c == 0 else "normal",
                )
            else:
                st.metric(
                    label="Найдено моделью",
                    value=f"{pred_pct:.1f}%  ({pred_n:,} пикс.)",
                )

    st.divider()

    debris_pct = 100 * int((mask == 0).sum()) / mask.size
    if has_gt and valid.sum() > 0:
        gt_debris_n   = int((gt_mask[valid] == 0).sum())
        gt_debris_pct = 100 * gt_debris_n / valid.sum()
        pred_n_valid  = mask[valid]
        gt_n_valid    = gt_mask[valid]
        tp = int(((pred_n_valid == 0) & (gt_n_valid == 0)).sum())
        fp = int(((pred_n_valid == 0) & (gt_n_valid != 0)).sum())
        fn = int(((pred_n_valid != 0) & (gt_n_valid == 0)).sum())
        prec = 100 * tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = 100 * tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        if gt_debris_n > 0:
            st.error(
                f"🗑️ **Мусор на снимке**: в датасете **{gt_debris_pct:.1f}%** ({gt_debris_n:,} пикс.)  "
                f"→  модель нашла **{debris_pct:.1f}%**  |  "
                f"Precision: **{prec:.1f}%**  |  Recall: **{rec:.1f}%**"
            )
        else:
            st.success(f"✅ Мусора в разметке нет. Модель пометила {debris_pct:.1f}% как мусор (ложные срабатывания).")
    else:
        if debris_pct > 1.0:
            st.error(f"⚠️ Модель обнаружила мусор: **{debris_pct:.1f}%** пикселей")
        else:
            st.success("✅ Значительного загрязнения не обнаружено")

       # Визуализация 
    fig = plot_results(rgb, mask, gt_mask)
    st.pyplot(fig, use_container_width=True)

    # Отчёт по классам 
    st.subheader("Отчёт по классам")

    has_gt = gt_mask is not None
    valid = gt_mask >= 0 if has_gt else None

    report_rows = []
    for c in range(4):
        name = TARGET_NAMES[c]
        pred_n = int((mask == c).sum())
        pred_pct = 100 * pred_n / mask.size

        if has_gt and valid.sum() > 0:
            gt_n = int((gt_mask[valid] == c).sum())
            gt_pct = 100 * gt_n / valid.sum()
            pred_on_valid = mask[valid]
            gt_on_valid   = gt_mask[valid]
            tp = int(((pred_on_valid == c) & (gt_on_valid == c)).sum())
            fp = int(((pred_on_valid == c) & (gt_on_valid != c)).sum())
            fn = int(((pred_on_valid != c) & (gt_on_valid == c)).sum())
            prec = 100 * tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            rec  = 100 * tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            report_rows.append({
                "Класс": name,
                "GT пикс.": f"{gt_n:,}",
                "GT %": f"{gt_pct:.1f}%",
                "Найдено пикс.": f"{pred_n:,}",
                "Найдено %": f"{pred_pct:.1f}%",
                "Precision": f"{prec:.1f}%" if not np.isnan(prec) else "—",
                "Recall": f"{rec:.1f}%" if not np.isnan(rec) else "—",
            })
        else:
            report_rows.append({
                "Класс": name,
                "Найдено пикс.": f"{pred_n:,}",
                "Найдено %": f"{pred_pct:.1f}%",
            })

    import pandas as pd
    df = pd.DataFrame(report_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if not has_gt:
        st.caption("ℹ️ Загрузите файл из `data/patches/` — тогда появятся колонки GT, Precision и Recall.")

    #  Сравнение с разметкой (если найдена) 
    if gt_mask is not None:
        valid = gt_mask >= 0
        if valid.sum() > 0:
            correct = (mask[valid] == gt_mask[valid]).sum()
            acc = correct / valid.sum() * 100
            labeled_pct = valid.sum() / gt_mask.size * 100
            st.info(
                f"📊 **Сравнение с разметкой**: "
                f"точность на размеченных пикселях = **{acc:.1f}%**  "
                f"({int(valid.sum()):,} пикс. из {gt_mask.size:,} имеют метку, {labeled_pct:.1f}%)"
            )
    else:
        st.caption("ℹ️ Файл разметки `_cl.tif` не найден рядом — Ground Truth не отображается.")

    #  Детальная статистика 
    with st.expander("Детальная статистика по пикселям"):
        for name, s in stats.items():
            st.write(f"**{name}**: {s['pixels']:,} пикс. ({s['pct']:.2f}%)")

    #  Скачать маску 
    buf = io.BytesIO()
    np.save(buf, mask)
    st.download_button(
        "💾 Скачать маску (.npy)",
        data=buf.getvalue(),
        file_name="classification_mask.npy",
        mime="application/octet-stream"
    )
else:
    st.info("Загрузите файл .tif — например, любой из `data/patches/*/`")

    with st.expander("Примеры доступных патчей"):
        patches_dir = Path(__file__).parent / "data" / "patches"
        if patches_dir.exists():
            for folder in sorted(patches_dir.iterdir())[:10]:
                tifs = list(folder.glob("*[0-9].tif"))  
                for t in tifs[:2]:
                    st.code(str(t.relative_to(Path(__file__).parent)))