import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import importlib
from scipy.io import loadmat
from matplotlib.patches import Patch
from PIL import Image

def compute_image_metrics(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    B_mean = np.mean(gray) / 255.0
    B_std  = np.std(gray) / 255.0

    R_mean, G_mean, B_mean_ch = cv2.mean(img)[:3]
    T_color = (R_mean - B_mean_ch) / (R_mean + B_mean_ch + 1e-5)
    Var_color = np.var(hsv[:, :, 0]) / 255.0
    L_var = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0
    return {
        "B_mean": round(B_mean, 4),
        "B_std":  round(B_std, 4),
        "T_color": round(T_color, 4),
        "Var_color": round(Var_color, 4),
        "L_var":  round(L_var, 4)
    }

def lighting_score(image_ids, df):
    def _compute_score(row):
        B_mean = row["B_mean"]
        B_std = row["B_std"]

        labels_artificial = ["light", "lamp", "chandelier", "sconce", "streetlight"]
        labels_natural = ["windowpane"]
        artificial = np.mean([row.get(lbl, 0) for lbl in labels_artificial])
        natural = np.mean([row.get(lbl, 0) for lbl in labels_natural])
        V_light = (0.4 * artificial + 0.6 * natural) / 100.0
        B_target = 0.7
        sigma = 0.18
        comfort = np.exp(-((B_mean - B_target) ** 2) / (2 * sigma ** 2))
        dark_penalty = 0.15 if B_mean < 0.45 else 0.0

        score = 0.7 * comfort + 0.5 * V_light - 0.15 * B_std + 0.1 - dark_penalty
        return np.clip(score, 0, 1)

    if isinstance(image_ids, str):
        image_ids = [image_ids]

    results = []
    for img_id in image_ids:
        row = df[df["image"] == img_id]
        if row.empty:
            print(f"Image {img_id} not found in dataset.")
            continue

        row = row.iloc[0]
        score_val = _compute_score(row)
        results.append({
            "image": img_id,
            "lighting_score": round(score_val * 100, 2)
        })

    return pd.DataFrame(results)

def visual_comfort_score(image_ids, df):
    def _compute_score(row):
        label_cols = [c for c in df.columns if c not in ["image", "B_mean", "B_std", "T_color", "Var_color", "L_var"]]
        values = np.array([row[c] for c in label_cols if row[c] > 0])
        num_labels = len(values)
        L_var = row["L_var"]
        optimal_num = 12
        clutter_curve = np.exp(-((num_labels - optimal_num) ** 2) / (2 * 5 ** 2))
        texture_penalty = min(L_var / 0.1, 1.0)
        if num_labels < 5 or num_labels > 30:
            clutter_curve *= 0.7

        comfort = (0.7 * clutter_curve + 0.3 * (1 - texture_penalty))
        return np.clip(comfort, 0, 1)

    if isinstance(image_ids, str):
        image_ids = [image_ids]

    results = []
    for img_id in image_ids:
        row = df[df["image"] == img_id]
        if row.empty:
            print(f"Image {img_id} not found.")
            continue
        row = row.iloc[0]
        score_val = _compute_score(row)
        results.append({
            "image": img_id,
            "visual_comfort_score": round(score_val * 100, 2)
        })

    return pd.DataFrame(results)

    
def affective_tone_score(image_ids, df):
    df = df.copy()
    df["B_mean_norm"] = (df["B_mean"] - df["B_mean"].min()) / (df["B_mean"].max() - df["B_mean"].min())
    df["T_color_norm"] = (df["T_color"] - df["T_color"].min()) / (df["T_color"].max() - df["T_color"].min())

    def sigmoid(x):
        return 1 / (1 + np.exp(-12 * (x - 0.5)))  # steeper response

    unified_labels = [
        "table", "coffee table", "pool table", "countertop", "desk", "chair", "cabinet", "shelf",
        "rug", "lamp", "light", "chandelier", "sconce", "curtain", "vase", "painting", "poster",
        "mirror", "clock", "pillow", "blanket", "cushion", "plant", "flower", "basket",
        "book", "tray", "sofa", "bed", "kitchen island", "stove", "sink", "refrigerator",
        "tree", "grass", "palm", "fence", "bench", "rock", "water", "road", "sky", "building"
    ]

    def _compute_score(row):
        B_mean = sigmoid(row["B_mean_norm"])
        T_color = sigmoid(row["T_color_norm"])
        B_std = row["B_std"]
        Var_color = row["Var_color"]
        decor_values = np.array([row.get(lbl, 0) for lbl in unified_labels])
        decor_present = np.sum(decor_values > 0)
        decor_strength = np.mean(decor_values[decor_values > 0]) / 10 if decor_present > 0 else 0
        decor_score = np.clip(0.65 * decor_strength + 0.35 * (decor_present / len(unified_labels)), 0, 1)
        indoor_labels = ["table", "chair", "lamp", "curtain", "painting", "plant", "cabinet", "windowpane", "floor"]
        indoor_bonus = min(np.sum([row.get(lbl, 0) > 0 for lbl in indoor_labels]) / 2.8, 1.0)
        barren_cues = (decor_present < 8) or (row["B_std"] < 0.12)
        barren_penalty = 0.5 if barren_cues else 0.0  # slightly stronger
        dark_penalty = 0.2 if row["B_mean"] < 0.45 else 0.0
        richness_pref = 4
        richness = np.exp(-((Var_color - richness_pref) ** 2) / (2 * 2.5 ** 2))

        score = (
            0.35 * B_mean +
            0.3 * T_color +
            0.15 * richness +
            0.1 * decor_score +
            0.1 * indoor_bonus -
            barren_penalty -
            dark_penalty
        )
        return np.clip(score, 0, 1)
    if isinstance(image_ids, str):
        image_ids = [image_ids]
    results = []
    for img_id in image_ids:
        row = df[df["image"] == img_id]
        if row.empty:
            print(f"Image {img_id} not found.")
            continue
        row = row.iloc[0]
        val = _compute_score(row)
        results.append({
            "image": img_id,
            "affective_tone_score": round(val * 100, 2)
        })

    return pd.DataFrame(results)

def aesthetic_enrichment_score(image_ids, df):
    unified_labels = [
        "painting", "poster", "sculpture", "vase", "mirror", "clock", "book", "tray", "basket",
        "curtain", "rug", "lamp", "chandelier", "sconce", "pillow", "blanket", "cushion",
        "plant", "flower", "tree", "grass", "palm", "rock", "water", "bench", "fence",
        "signboard", "flag", "fountain", "awning", "stage", "table", "coffee table", "pool table",
        "desk", "countertop", "chair", "cabinet", "sofa", "bed", "kitchen island", "stove", "sink",
        "refrigerator", "windowpane", "floor", "ceiling"
    ]
    def _compute_score(row):
        Var_color = row["Var_color"]
        B_std = row["B_std"]
        present_values = np.array([row.get(lbl, 0) for lbl in unified_labels])
        present_labels = present_values[present_values > 0]
        decor_density = np.mean(present_labels) / 10 if len(present_labels) > 0 else 0
        decor_variety = len(present_labels) / len(unified_labels)
        decor_score = np.clip(0.7 * decor_variety + 0.3 * decor_density, 0, 1)
        color_richness = np.exp(-((Var_color - 6) ** 2) / (2 * 3 ** 2))
        harmony = np.clip(1 - B_std * 2.2, 0, 1)
        diversity = np.clip(len(present_labels) / 25, 0, 1)
        indoor_labels = ["table", "chair", "lamp", "curtain", "painting", "plant", "cabinet", "windowpane", "floor"]
        indoor_bonus = min(np.sum([row.get(lbl, 0) > 0 for lbl in indoor_labels]) / 2.8, 1.0)
        barren_cues = (len(present_labels) < 8) or (row["B_std"] < 0.12)
        barren_penalty = 0.55 if barren_cues else 0.0  # stronger penalty

        score = (
            0.45 * decor_score +
            0.25 * color_richness +
            0.15 * harmony +
            0.1 * diversity +
            0.05 * indoor_bonus -
            barren_penalty
        )
        return np.clip(score, 0.1, 1)  # floor at 10%

    if isinstance(image_ids, str):
        image_ids = [image_ids]

    results = []
    for img_id in image_ids:
        row = df[df["image"] == img_id]
        if row.empty:
            print(f"Image {img_id} not found.")
            continue
        row = row.iloc[0]
        val = _compute_score(row)
        results.append({
            "image": img_id,
            "aesthetic_enrichment_score": round(val * 100, 2)
        })

    return pd.DataFrame(results)

#only part to touch 
def compute_total_score(df_scores):
    weights = {
        "lighting_score": 0.20,
        "visual_comfort_score": 0.40,
        "affective_tone_score": 0.10,
        "aesthetic_enrichment_score": 0.30
    }

    df_scores["total_score"] = (
        df_scores["lighting_score"] * weights["lighting_score"] +
        df_scores["visual_comfort_score"] * weights["visual_comfort_score"] +
        df_scores["affective_tone_score"] * weights["affective_tone_score"] +
        df_scores["aesthetic_enrichment_score"] * weights["aesthetic_enrichment_score"]
    ).round(2)

    df_scores["total_score"] = (100 * (df_scores["total_score"] / 100) ** 0.75).round(2)
    df_scores["total_score"] = np.where(df_scores["total_score"] < 50,
                                        df_scores["total_score"] * 1.02,
                                        df_scores["total_score"] * 1.1)
    df_scores["total_score"] = np.clip(df_scores["total_score"], 0, 100)
    def interpret_score(x):
        if x >= 80:
            return "This is an excellent space!"
        elif x >= 65:
            return "This is a good space!"
        elif x >= 50:
            return "This place is alright..."
        else:
            return "This place needs some serious improvement"

    df_scores["rating"] = df_scores["total_score"].apply(interpret_score)
    return df_scores

label_names = [
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass',
    'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair',
    'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field',
    'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion',
    'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace',
    'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway',
    'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench',
    'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel',
    'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
    'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet',
    'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool',
    'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball',
    'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher',
    'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan',
    'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag'
]

def visualize_top_images(
    df_scores,
    images_path,
    masks_path,
    color_mat_path,
    label_names,
    score_types=None,
    top_k=5
):
    """
    Drop-in replacement:
    - For each score in `score_types`, plot a line chart of ALL images' scores in
      the order they appear in `images_path` (natural filename order).
    - Attach thumbnails above the line for the top-k highest scores.
    - Signature and inputs are identical to the original function.
    """
    import os, re
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    # Keep compatibility with original: allow score_types to be None or str
    all_scores = [
        "lighting_score",
        "visual_comfort_score",
        "affective_tone_score",
        "aesthetic_enrichment_score",
        "total_score",
    ]
    if score_types is None:
        score_types = all_scores
    elif isinstance(score_types, str):
        score_types = [score_types]

    # Helper: natural sort so img2 < img10
    def natural_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

    # Build folder sequence (image stems in folder order)
    IMG_EXTS = (".jpg", ".jpeg", ".png")
    folder_files = [f for f in os.listdir(images_path) if f.lower().endswith(IMG_EXTS)]
    folder_files.sort(key=natural_key)

    # Map image_id (without extension) -> best existing path (jpg preferred, else png)
    def pick_path(stem):
        p_jpg = os.path.join(images_path, f"{stem}.jpg")
        p_png = os.path.join(images_path, f"{stem}.png")
        if os.path.exists(p_jpg): return p_jpg
        if os.path.exists(p_png): return p_png
        return None

    folder_stems = [os.path.splitext(f)[0] for f in folder_files]
    folder_paths = [pick_path(stem) for stem in folder_stems]

    # Prepare a lookup from df_scores
    # Expect df_scores["image"] to match the stems (str)
    df_scores = df_scores.copy()
    df_scores["image"] = df_scores["image"].astype(str)
    id_to_row = {str(row["image"]): row for _, row in df_scores.iterrows()}

    # For each requested score column, build y-values in folder sequence and plot
    for col in score_types:
        if col not in df_scores.columns:
            print(f"Column '{col}' not found in df_scores.")
            continue

        # Build (score, img_path) in strict folder order; skip items without data or file
        seq_scores = []
        seq_paths = []
        seq_indices = []
        for i, (stem, path) in enumerate(zip(folder_stems, folder_paths)):
            if path is None:
                continue
            row = id_to_row.get(stem)
            if row is None:
                continue
            val = row[col]
            # must be numeric
            try:
                y = float(val)
            except Exception:
                continue
            seq_scores.append(y)
            seq_paths.append(path)
            seq_indices.append(i)

        if len(seq_scores) == 0:
            print(f"No plottable data for '{col}'.")
            continue

        # X positions: normalized [0,1] across the plotted items (keep sequence spacing uniform)
        N = len(seq_scores)
        xs = np.linspace(0, 1, N) if N > 1 else np.array([0.5])
        ys = np.array(seq_scores, dtype=float)

        # Determine top-k by score
        k = min(top_k, N)
        top_idx = np.argsort(-ys)[:k]  # indices in the plotted sequence

        # Axis ranges and padding
        ymin, ymax = float(ys.min()), float(ys.max())
        yr = (ymax - ymin) if ymax > ymin else 1.0
        pad = 0.12 * yr
        top_extra = 2.5 * pad

        # Figure per score type (mirrors original behavior of one fig per col)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(ymin - pad, ymax + top_extra)

        # Line + points
        # Draw line and points ABOVE thumbnails
        ax.plot(xs, ys, color="black", linewidth=1.5, alpha=0.7, zorder=4)
        ax.scatter(xs, ys, color="black", s=10, zorder=5)


        # Thumbnails for top-k (placed above the line)
        thumb_y = ymax + 1.3 * pad
        offsets = np.linspace(-0.1 * pad, 0.1 * pad, num=min(k, 5)) if k > 1 else [0.0]
        for j, i_top in enumerate(sorted(top_idx, key=lambda t: xs[t])):  # left→right
            img_path = seq_paths[i_top]
            im_bgr = cv2.imread(img_path)
            if im_bgr is None:
                continue
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            oi = OffsetImage(im_rgb, zoom=0.1, resample=True)

            dy = offsets[j % len(offsets)]
            ab = AnnotationBbox(
    oi, (xs[i_top], thumb_y + dy),
    frameon=False, box_alignment=(0.5, 1.0),
    clip_on=False,
    zorder=1        # 👈 add this line
)
            ax.add_artist(ab)
            ax.add_artist(ab)

            # Optional numeric label above thumbnail
            ax.text(xs[i_top], thumb_y + dy + 0.02 * yr, f"{ys[i_top]:.2f}",
                    ha="center", va="bottom", fontsize=9, zorder=6)

        # Titles/labels to match your style
        ax.set_title(f'Images {col.replace("_", " ").title()} over timeline (sequence order)', fontsize=16)
        ax.set_xlabel("Frame Index (normalized)")
        ax.set_ylabel(col.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()



