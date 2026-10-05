import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
import io
import math

st.set_page_config(
    page_title="PredLatent-Track Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

_FONT_CACHE = {}


def get_font(size):
    if size not in _FONT_CACHE:
        font = None
        for path in (
            "DejaVuSans.ttf",
            "DejaVuSans-Bold.ttf",
            "Arial.ttf",
            "LiberationSans-Regular.ttf",
        ):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                pass
        if font is None:
            font = ImageFont.load_default()
        _FONT_CACHE[size] = font
    return _FONT_CACHE[size]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def lerp(a, b, t):
    return a + (b - a) * t


def ease_in_out(t):
    t = clamp(float(t), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def stable_hash(value):
    s = str(value)
    return sum((i + 1) * ord(c) for i, c in enumerate(s)) % 100000


TRACK_COLORS = {
    1: (86, 199, 255),
    2: (255, 85, 150),
    3: (255, 190, 60),
}


def draw_grid(draw, ox, oy, w, h):
    for x in range(ox, ox + w, 40):
        draw.line((x, oy, x, oy + h), fill=(24, 32, 52), width=1)
    for y in range(oy, oy + h, 40):
        draw.line((ox, y, ox + w, y), fill=(24, 32, 52), width=1)


def draw_object(draw, ox, oy, w, h, x, y, track_id, note="", swap=False):
    cx = ox + int(x * w)
    cy = oy + int(y * h)
    r = int(0.055 * w)

    color = TRACK_COLORS.get(int(track_id), (220, 220, 220))
    if swap:
        color = (255, 80, 80)

    fill = (color[0] // 4, color[1] // 4, color[2] // 4)

    draw.ellipse(
        (cx - r, cy - r, cx + r, cy + r),
        fill=fill,
        outline=color,
        width=3,
    )
    draw.text(
        (cx - 5, cy - 8),
        str(int(track_id)),
        fill=(245, 248, 255),
        font=get_font(14),
    )
    if note:
        draw.text(
            (cx - r, cy + r + 2),
            note,
            fill=color,
            font=get_font(11),
        )


def simulate_scene(
    fps,
    duration,
    occ_len,
    horizon,
    slots,
    enable_predictor,
    enable_vicreg,
    show_ghosts,
    seed,
):
    rng = np.random.default_rng(int(seed))

    n_frames = max(18, int(round(float(fps) * float(duration))))
    occ_len = int(clamp(occ_len, 4, max(4, n_frames - 6)))

    occ_start = max(3, n_frames // 2 - occ_len // 2)
    occ_end = min(n_frames - 3, occ_start + occ_len)
    if occ_end <= occ_start:
        occ_end = occ_start + 1

    width = 512
    panel_height = 288

    occ_rect = (0.33, 0.16, 0.67, 0.84)

    tau_pred = 5.0 + 2.2 * float(horizon) + 0.18 * float(slots)
    if enable_vicreg:
        tau_pred += 7.0
    tau_base = 3.5

    if not enable_predictor:
        pred_keeps = False
    else:
        score = 0.44 + 0.055 * float(horizon) + 0.009 * float(slots)
        if enable_vicreg:
            score += 0.14
        score *= math.exp(-max(0, occ_len - 30) / 95.0)
        if occ_len <= 15:
            score += 0.08
        score = clamp(score, 0.04, 0.97)
        pred_keeps = bool(rng.random() < score)

    exit_lost = max(0, occ_end - occ_start)
    if enable_predictor:
        pred_conf_exit = math.exp(-exit_lost / tau_pred)
    else:
        pred_conf_exit = math.exp(-exit_lost / tau_base)

    if pred_conf_exit < 0.09:
        pred_keeps = False

    baseline_swaps = occ_len > 12

    ph1 = float(rng.uniform(0, 2 * math.pi))
    ph2 = float(rng.uniform(0, 2 * math.pi))

    positions = []
    for t in range(n_frames):
        p = t / (n_frames - 1) if n_frames > 1 else 0.0
        e = ease_in_out(p)

        x1 = clamp(
            lerp(0.13, 0.87, e) + 0.012 * math.sin(5.0 * p + ph1),
            0.03,
            0.97,
        )
        y1 = clamp(0.58 + 0.055 * math.sin(3.0 * p + ph1), 0.12, 0.88)

        x2 = clamp(
            lerp(0.87, 0.13, e) + 0.012 * math.sin(4.0 * p + ph2),
            0.03,
            0.97,
        )
        y2 = clamp(0.42 + 0.055 * math.cos(3.4 * p + ph2), 0.12, 0.88)

        positions.append(((x1, y1), (x2, y2)))

    def visible(oid, t):
        x, _ = positions[t][oid - 1]
        return not (
            occ_start <= t < occ_end
            and occ_rect[0] < x < occ_rect[2]
        )

    last_seen_base = {
        1: positions[0][0],
        2: positions[0][1],
    }
    last_seen_pred = {
        1: positions[0][0],
        2: positions[0][1],
    }

    velocity = {
        1: (0.0, 0.0),
        2: (0.0, 0.0),
    }

    prev_base = {1: None, 2: None}
    prev_pred = {1: None, 2: None}

    base_idsw = 0
    pred_idsw = 0

    frames = []
    chart_rows = []

    entry_t = max(0, occ_start - 1)

    for t in range(n_frames):
        pos = {
            1: positions[t][0],
            2: positions[t][1],
        }
        vis = {
            1: visible(1, t),
            2: visible(2, t),
        }

        if t == entry_t and entry_t > 0:
            for oid in (1, 2):
                px, py = positions[entry_t - 1][oid - 1]
                cx, cy = positions[entry_t][oid - 1]
                velocity[oid] = (cx - px, cy - py)

        if t < occ_start or t >= occ_end:
            lost = 0
        else:
            lost = t - entry_t

        conf_base = math.exp(-lost / tau_base)
        if enable_predictor:
            conf_pred = math.exp(-lost / tau_pred)
        else:
            conf_pred = math.exp(-lost / tau_base)

        assigns_base = {}
        assigns_pred = {}
        ghosts = []

        det_ids = [oid for oid in (1, 2) if vis[oid]]

        # Baseline: appearance / short-term motion association proxy.
        if det_ids:
            if t < occ_start or not baseline_swaps:
                for oid in det_ids:
                    assigns_base[oid] = oid
            else:
                used = set()
                dets = []
                for oid in det_ids:
                    d1 = dist(pos[oid], last_seen_base[1])
                    d2 = dist(pos[oid], last_seen_base[2])
                    dets.append((oid, d1, d2))

                for oid, d1, d2 in sorted(dets, key=lambda z: min(z[1], z[2])):
                    cand = 1 if d1 <= d2 else 2
                    if cand in used:
                        cand = 2 if cand == 1 else 1
                    assigns_base[oid] = cand
                    used.add(cand)

            for oid, tid in assigns_base.items():
                last_seen_base[tid] = pos[oid]

            for oid in det_ids:
                cur = assigns_base.get(oid)
                if cur is not None:
                    if prev_base[oid] is not None and cur != prev_base[oid]:
                        base_idsw += 1
                    prev_base[oid] = cur

        # PredLatent: latent-prediction association proxy.
        if det_ids:
            if t < occ_start or pred_keeps:
                for oid in det_ids:
                    assigns_pred[oid] = oid
            else:
                used = set()
                dets = []
                for oid in det_ids:
                    d1 = dist(pos[oid], last_seen_pred[1])
                    d2 = dist(pos[oid], last_seen_pred[2])
                    dets.append((oid, d1, d2))

                for oid, d1, d2 in sorted(dets, key=lambda z: min(z[1], z[2])):
                    cand = 1 if d1 <= d2 else 2
                    if cand in used:
                        cand = 2 if cand == 1 else 1
                    assigns_pred[oid] = cand
                    used.add(cand)

            for oid, tid in assigns_pred.items():
                last_seen_pred[tid] = pos[oid]

            for oid in det_ids:
                cur = assigns_pred.get(oid)
                if cur is not None:
                    if prev_pred[oid] is not None and cur != prev_pred[oid]:
                        pred_idsw += 1
                    prev_pred[oid] = cur

        # Predicted-latent ghosts during occlusion.
        if show_ghosts and occ_start <= t < occ_end:
            for tid in (1, 2):
                if vis[tid]:
                    continue

                if enable_predictor:
                    if pred_keeps:
                        gx, gy = pos[tid]
                        noise_scale = 0.035 * (1.0 - conf_pred)
                        gx = clamp(
                            gx + float(rng.normal(0.0, noise_scale + 1e-6)),
                            0.03,
                            0.97,
                        )
                        gy = clamp(
                            gy + float(rng.normal(0.0, noise_scale + 1e-6)),
                            0.08,
                            0.92,
                        )
                    else:
                        gx, gy = last_seen_pred[tid]
                else:
                    gx, gy = last_seen_pred[tid]
                    conf_pred = conf_pred * 0.45

                ghosts.append(
                    {
                        "id": tid,
                        "x": float(gx),
                        "y": float(gy),
                        "conf": float(conf_pred),
                    }
                )

        frames.append(
            {
                "t": t,
                "pos": pos,
                "vis": vis,
                "assigns_base": assigns_base,
                "assigns_pred": assigns_pred,
                "ghosts": ghosts,
                "conf_base": float(conf_base),
                "conf_pred": float(conf_pred),
                "base_idsw": int(base_idsw),
                "pred_idsw": int(pred_idsw),
                "occ_interval": (int(occ_start), int(occ_end)),
                "occ_rect": occ_rect,
            }
        )

        chart_rows.append(
            {
                "frame": t,
                "time_s": t / float(fps),
                "Baseline confidence": float(conf_base),
                "Pred confidence": float(conf_pred),
                "Baseline IDSW": int(base_idsw),
                "PredLatent IDSW": int(pred_idsw),
            }
        )

    meta = {
        "width": width,
        "panel_height": panel_height,
        "n_frames": n_frames,
        "fps": int(fps),
        "occ_start": int(occ_start),
        "occ_end": int(occ_end),
        "tau_pred": float(tau_pred),
        "tau_base": float(tau_base),
    }

    summary = {
        "base_idsw": int(base_idsw),
        "pred_idsw": int(pred_idsw),
        "pred_keeps": bool(pred_keeps),
        "baseline_swaps": bool(baseline_swaps),
        "occ_frames": int(occ_end - occ_start),
        "pred_conf_exit": float(pred_conf_exit),
    }

    return {
        "frames": frames,
        "meta": meta,
        "chart": pd.DataFrame(chart_rows),
        "summary": summary,
    }


def render_frame(frame, meta, panel_mode):
    w = meta["width"]
    ph = meta["panel_height"]
    header = 30

    if panel_mode == "Both panels":
        h = ph * 2 + header
        offsets = [header, header + ph]
        titles = [
            "ByteTrack-like baseline (simulated)",
            "PredLatent-Track (simulated)",
        ]
    elif panel_mode == "PredLatent only":
        h = ph
        offsets = [0]
        titles = ["PredLatent-Track (simulated)"]
    else:
        h = ph
        offsets = [0]
        titles = ["ByteTrack-like baseline (simulated)"]

    img = Image.new("RGB", (w, h), (8, 12, 24))
    draw = ImageDraw.Draw(img)

    if panel_mode == "Both panels":
        draw.text(
            (10, 6),
            "PredLatent-Track | synthetic crossing-occlusion demo",
            fill=(220, 230, 255),
            font=get_font(14),
        )

    for ox, title in zip(offsets, titles):
        draw.rectangle((0, ox, w, ox + ph), fill=(12, 18, 34), outline=(45, 58, 90), width=1)
        draw_grid(draw, 0, ox, w, ph)
        draw.text((8, ox + 6), title, fill=(190, 205, 235), font=get_font(12))

        occ = frame["occ_rect"]
        ox0 = int(occ[0] * w)
        oy0 = ox + int(occ[1] * ph)
        ox1 = int(occ[2] * w)
        oy1 = ox + int(occ[3] * ph)

        draw.rectangle(
            (ox0, oy0, ox1, oy1),
            fill=(32, 40, 64),
            outline=(105, 120, 160),
            width=2,
        )
        draw.text(
            (ox0 + 6, oy0 + 6),
            "occluder",
            fill=(150, 165, 205),
            font=get_font(10),
        )

        is_pred = "Pred" in title

        if is_pred:
            for g in frame["ghosts"]:
                color = TRACK_COLORS.get(int(g["id"]), (220, 220, 220))
                cx = int(g["x"] * w)
                cy = ox + int(g["y"] * ph)
                r = int(0.045 * w)

                draw.ellipse(
                    (cx - r, cy - r, cx + r, cy + r),
                    outline=color,
                    width=2,
                )
                draw.text(
                    (cx - 5, cy - 7),
                    "?",
                    fill=color,
                    font=get_font(13),
                )
                draw.ellipse(
                    (cx - 2, cy - 2, cx + 2, cy + 2),
                    fill=color,
                )

        for oid in (1, 2):
            if not frame["vis"][oid]:
                continue

            x, y = frame["pos"][oid]

            if is_pred:
                tid = frame["assigns_pred"].get(oid, oid)
                swap = tid != oid and frame["t"] >= frame["occ_interval"][1]
                draw_object(
                    draw,
                    0,
                    ox,
                    w,
                    ph,
                    x,
                    y,
                    tid,
                    note="IDSW" if swap else "",
                    swap=swap,
                )
            else:
                tid = frame["assigns_base"].get(oid, oid)
                swap = tid != oid and frame["t"] >= frame["occ_interval"][1]
                draw_object(
                    draw,
                    0,
                    ox,
                    w,
                    ph,
                    x,
                    y,
                    tid,
                    note="IDSW" if swap else "",
                    swap=swap,
                )

        conf = frame["conf_pred"] if is_pred else frame["conf_base"]
        idsw = frame["pred_idsw"] if is_pred else frame["base_idsw"]

        hud_y = ox + ph - 22
        draw.rectangle(
            (6, hud_y, 170, hud_y + 16),
            fill=(20, 28, 48),
            outline=(60, 75, 110),
        )

        bar_w = int(120 * clamp(conf, 0.0, 1.0))
        if conf > 0.35:
            bar_color = (80, 220, 140)
        elif conf > 0.18:
            bar_color = (230, 180, 60)
        else:
            bar_color = (235, 80, 80)

        draw.rectangle(
            (8, hud_y + 2, 8 + bar_w, hud_y + 14),
            fill=bar_color,
        )
        draw.text(
            (132, hud_y - 1),
            f"conf {conf:.2f}",
            fill=(210, 220, 240),
            font=get_font(10),
        )

        draw.text(
            (w - 120, ox + 6),
            f"frame {frame['t'] + 1}/{meta['n_frames']}",
            fill=(180, 195, 225),
            font=get_font(10),
        )
        draw.text(
            (w - 150, hud_y),
            f"ID switches: {idsw}",
            fill=(255, 210, 120) if idsw else (130, 240, 170),
            font=get_font(11),
        )

    return img


def render_scene(scene, panel_mode):
    return [render_frame(f, scene["meta"], panel_mode) for f in scene["frames"]]


def encode_gif(frames, fps):
    if not frames:
        return b""

    buf = io.BytesIO()
    duration = max(33, int(1000 / max(1, int(fps))))

    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def build_demo(
    fps,
    duration,
    occ_len,
    horizon,
    slots,
    enable_predictor,
    enable_vicreg,
    show_ghosts,
    seed,
    panel_mode,
):
    scene = simulate_scene(
        fps=fps,
        duration=duration,
        occ_len=occ_len,
        horizon=horizon,
        slots=slots,
        enable_predictor=enable_predictor,
        enable_vicreg=enable_vicreg,
        show_ghosts=show_ghosts,
        seed=seed,
    )

    pil_frames = render_scene(scene, panel_mode)
    gif = encode_gif(pil_frames, fps)

    return {
        "gif": gif,
        "chart": scene["chart"],
        "meta": scene["meta"],
        "summary": scene["summary"],
    }


@st.cache_data(show_spinner=False)
def build_benchmark(seed, horizon, slots, enable_predictor, enable_vicreg):
    methods = [
        "ByteTrack",
        "OC-SORT",
        "BoT-SORT",
        "SAM 2 video",
        "PredLatent-Track (sim)",
    ]

    buckets = [
        ("1-10", 6),
        ("11-30", 20),
        (">30", 55),
    ]

    baseline_cfg = {
        "ByteTrack": (0.0, 0.0),
        "OC-SORT": (1.3, 1.6),
        "BoT-SORT": (2.0, 2.4),
        "SAM 2 video": (2.7, 3.3),
    }

    rows = []

    for method in methods:
        for bucket, avg_occ in buckets:
            hotas = []
            assas = []
            idf1s = []
            idsws = []

            for s in range(3):
                rng = np.random.default_rng(
                    int(seed) * 1000 + s * 31 + stable_hash(method + bucket)
                )

                if method.startswith("Pred"):
                    if enable_predictor:
                        h_eff = float(horizon)
                        sl_eff = float(slots)
                        v_eff = 1.0 if enable_vicreg else 0.0
                    else:
                        h_eff = 0.0
                        sl_eff = float(slots) * 0.35
                        v_eff = 0.0

                    idsw = (
                        25.0
                        - 1.35 * h_eff
                        - 0.18 * sl_eff
                        - 5.2 * v_eff
                        + 0.15 * max(0, avg_occ - 18)
                        + rng.normal(0, 1.6)
                    )
                    hota = (
                        75.0
                        + 0.60 * h_eff
                        + 0.070 * sl_eff
                        + 2.5 * v_eff
                        - 0.070 * max(0, avg_occ - 18)
                        + rng.normal(0, 1.0)
                    )
                else:
                    idsw_bonus, hota_bonus = baseline_cfg[method]
                    idsw = (
                        31.0
                        - 2.0 * idsw_bonus
                        + 0.30 * max(0, avg_occ - 8)
                        + rng.normal(0, 2.1)
                    )
                    hota = (
                        68.0
                        + 1.5 * hota_bonus
                        - 0.085 * max(0, avg_occ - 8)
                        + rng.normal(0, 1.3)
                    )

                idsw = max(0.0, float(idsw))
                hota = float(np.clip(hota, 28.0, 96.0))
                assa = float(
                    np.clip(
                        hota - 2.2 - 0.035 * idsw + rng.normal(0, 0.8),
                        22.0,
                        95.0,
                    )
                )
                idf1 = float(
                    np.clip(
                        hota + 1.6 - 0.018 * idsw + rng.normal(0, 1.1),
                        24.0,
                        97.0,
                    )
                )

                hotas.append(hota)
                assas.append(assa)
                idf1s.append(idf1)
                idsws.append(idsw)

            rows.append(
                {
                    "Method": method,
                    "Occlusion bucket": bucket,
                    "HOTA mean": float(np.mean(hotas)),
                    "HOTA std": float(np.std(hotas, ddof=1)),
                    "AssA mean": float(np.mean(assas)),
                    "AssA std": float(np.std(assas, ddof=1)),
                    "IDF1 mean": float(np.mean(idf1s)),
                    "IDF1 std": float(np.std(idf1s, ddof=1)),
                    "IDSW / 100 tracks mean": float(np.mean(idsws)),
                    "IDSW / 100 tracks std": float(np.std(idsws, ddof=1)),
                }
            )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_ablation(seed):
    variants = [
        ("No predictor", 0, 12, False, True, "frozen"),
        ("Horizon 1", 1, 12, True, True, "frozen"),
        ("Horizon 4", 4, 12, True, True, "frozen"),
        ("Horizon 8", 8, 12, True, True, "frozen"),
        ("Slots 6", 4, 6, True, True, "frozen"),
        ("Slots 12", 4, 12, True, True, "frozen"),
        ("Slots 24", 4, 24, True, True, "frozen"),
        ("Frozen encoder", 4, 12, True, True, "frozen"),
        ("Fine-tuned encoder", 4, 12, True, True, "fine-tuned"),
        ("VICReg off", 4, 12, True, False, "frozen"),
        ("Full PredLatent (sim)", 4, 12, True, True, "frozen"),
    ]

    rows = []

    for name, h, sl, pred, vic, enc in variants:
        hotas = []
        idsws = []
        vars_ = []
        eranks = []

        for s in range(3):
            rng = np.random.default_rng(
                int(seed) * 1000 + s * 17 + stable_hash(name)
            )

            enc_bonus = 1.4 if enc == "fine-tuned" else 0.0
            vic_flag = 1.0 if vic else 0.0
            collapse = 0.0 if vic else 1.0

            if pred:
                h_eff = float(h)
                sl_eff = float(sl)
            else:
                h_eff = 0.0
                sl_eff = float(sl) * 0.45

            idsw = (
                27.0
                - 1.3 * h_eff
                - 0.17 * sl_eff
                - 5.0 * vic_flag
                + 3.2 * collapse
                - enc_bonus
                + rng.normal(0, 1.5)
            )

            hota = (
                73.0
                + 0.55 * h_eff
                + 0.065 * sl_eff
                + 2.2 * vic_flag
                + enc_bonus
                - 2.0 * collapse
                + rng.normal(0, 1.0)
            )

            slot_var = (
                0.26
                + 0.05 * vic_flag
                + 0.002 * sl_eff
                - 0.16 * collapse
                - 0.05 * (0.0 if pred else 1.0)
                + rng.normal(0, 0.02)
            )

            effective_rank = (
                2.0
                + 0.16 * sl_eff
                + 2.2 * vic_flag
                + 0.25 * h_eff
                - 4.5 * collapse
                + rng.normal(0, 0.35)
            )

            hotas.append(float(np.clip(hota, 25.0, 96.0)))
            idsws.append(max(0.0, float(idsw)))
            vars_.append(float(np.clip(slot_var, 0.01, 0.90)))
            eranks.append(float(np.clip(effective_rank, 1.0, float(sl))))

        rows.append(
            {
                "Variant": name,
                "HOTA >30 mean": float(np.mean(hotas)),
                "HOTA >30 std": float(np.std(hotas, ddof=1)),
                "IDSW / 100 mean": float(np.mean(idsws)),
                "IDSW / 100 std": float(np.std(idsws, ddof=1)),
                "Slot variance": float(np.mean(vars_)),
                "Effective rank": float(np.mean(eranks)),
            }
        )

    return pd.DataFrame(rows)


def build_failures():
    data = [
        (
            "F01",
            "Two similar dancers cross behind a banner for 45 frames",
            "Baseline swaps IDs; PredLatent confidence decays but keeps IDs",
            "Long occlusion with similar appearance embeddings",
            "Increase horizon to 8; use EMA target consistency",
            18.0,
        ),
        (
            "F02",
            "12 players, only 6 slots allocated",
            "Two objects share one slot; identity binding collapses",
            "Insufficient slot capacity",
            "Increase slots to 12 or 24; add variance regularization",
            10.0,
        ),
        (
            "F03",
            "SportsMOT-style players wearing same jersey color",
            "Appearance-only matcher confuses identities",
            "Color and texture are not discriminative",
            "Rely on predicted latent dynamics, not appearance cosine alone",
            16.0,
        ),
        (
            "F04",
            "Object fully occluded for 90 frames",
            "Predictor confidence falls below learned lost threshold",
            "Extrapolation uncertainty grows too large",
            "Adaptive lost threshold; re-initialize from nearby slot if confident",
            22.0,
        ),
        (
            "F05",
            "Fast camera pan plus object acceleration",
            "Predicted box drifts away from observed reappearance",
            "Linear latent dynamics under-models abrupt motion",
            "Add cross-slot attention and motion-conditioned predictor",
            12.0,
        ),
        (
            "F06",
            "New object enters while another exits",
            "Slot initialization briefly binds wrong identity",
            "Temporal slot carry-over lag",
            "SAVi-style slot init from previous frame plus birth/death heads",
            14.0,
        ),
        (
            "F07",
            "Crowded crossing with three similar objects",
            "One slot attends to two objects intermittently",
            "Cross-slot interference and competition",
            "Explicit interaction attention and slot exclusivity loss",
            11.0,
        ),
        (
            "F08",
            "Low-texture mug on plain table",
            "Frozen DINOv2 patch features are weakly discriminative",
            "Patch embeddings lack object-specific detail",
            "Multi-scale features or light adapter tuning",
            13.0,
        ),
        (
            "F09",
            "Webcam demo with two identical colored cups",
            "Color-based heuristic suggests same identity",
            "Visual similarity dominates naive association",
            "Use latent prediction and trajectory prior as primary cue",
            17.0,
        ),
        (
            "F10",
            "Hungarian matching threshold too aggressive",
            "Correct prediction rejected as lost track",
            "Fixed threshold does not account for uncertainty decay",
            "Learned lost threshold conditioned on confidence and occlusion length",
            15.0,
        ),
    ]

    return pd.DataFrame(
        data,
        columns=[
            "Case",
            "Scenario",
            "Observed failure",
            "Root cause",
            "Mitigation",
            "Confidence tau",
        ],
    )


# -----------------------------
# Streamlit UI
# -----------------------------

st.sidebar.title("PredLatent-Track controls")
st.sidebar.caption("Change any control; the demo regenerates immediately.")

seed = int(st.sidebar.number_input("Random seed", 0, 9999, 7, step=1))
duration = int(st.sidebar.slider("Demo length (seconds)", 3, 8, 6, step=1))
fps = int(st.sidebar.selectbox("Animation FPS", [10, 12, 15], index=2))
occ_len = int(st.sidebar.slider("Occlusion length (frames)", 5, 120, 45, step=1))
horizon = int(st.sidebar.slider("Predictor horizon", 1, 8, 4, step=1))
slots = int(st.sidebar.selectbox("Object slots", [6, 12, 24], index=1))

enable_predictor = st.sidebar.checkbox("Enable latent predictor", value=True)
enable_vicreg = st.sidebar.checkbox("VICReg anti-collapse", value=True)
show_ghosts = st.sidebar.checkbox("Show predicted latent ghosts", value=True)

panel_mode = st.sidebar.radio(
    "View",
    ["Both panels", "PredLatent only", "Baseline only"],
    index=0,
)

if st.sidebar.button("Reset cached demos"):
    st.cache_data.clear()
    st.rerun()

st.title("PredLatent-Track: Slot-Latent Prediction for Multi-Object Tracking")
st.caption(
    "Self-contained Streamlit proof-of-concept. All pixels, tracks, and metrics below are "
    "synthetic simulator outputs, not real DINOv2 / JEPA / Slot-Attention inference or "
    "benchmark results. The purpose is to make the research hypothesis interactive and testable."
)

with st.spinner("Loading simulated demo and starting playback..."):
    demo = build_demo(
        fps=fps,
        duration=duration,
        occ_len=occ_len,
        horizon=horizon,
        slots=slots,
        enable_predictor=enable_predictor,
        enable_vicreg=enable_vicreg,
        show_ghosts=show_ghosts,
        seed=seed,
        panel_mode=panel_mode,
    )

col_video, col_side = st.columns([2.1, 1])

with col_video:
    st.subheader("Auto-playing simulated demo")
    st.image(demo["gif"], use_container_width=True)
    st.download_button(
        "Download demo reel (GIF)",
        data=demo["gif"],
        file_name="predlatent_track_demo.gif",
        mime="image/gif",
    )

with col_side:
    st.subheader("Live-style diagnostics")

    m1, m2 = st.columns(2)
    m1.metric("Baseline IDSW", demo["summary"]["base_idsw"])
    m2.metric("PredLatent IDSW", demo["summary"]["pred_idsw"])

    st.metric(
        "Predictor kept IDs?",
        "Yes" if demo["summary"]["pred_keeps"] else "No / ablated",
    )
    st.metric("Occlusion frames", demo["summary"]["occ_frames"])

    conf_df = demo["chart"].set_index("frame")[
        ["Baseline confidence", "Pred confidence"]
    ]
    st.line_chart(conf_df)

tab_notes, tab_bench, tab_ablation, tab_fail, tab_model = st.tabs(
    [
        "Experiment notes",
        "Benchmark summary",
        "Ablation table",
        "Failure gallery",
        "Model card",
    ]
)

with tab_notes:
    st.markdown(
        """
        ### Research question

        Can a label-free, slot-based latent dynamics model retain identities through long
        occlusions better than appearance-only association?

        ### Hypothesis

        Predicting the next-step latent of each object slot for horizons 1–8 frames gives a
        motion-and-appearance prior that can reduce ID switches in occlusions longer than
        30 frames versus appearance-only baselines.

        ### What this app actually does

        This is a deterministic synthetic micro-simulator. It generates a two-object crossing
        scenario, hides the objects behind an occluder, and compares:

        - a baseline association proxy that relies on last-seen position / appearance similarity;
        - a PredLatent proxy that rolls slot latents and boxes forward during occlusion.

        It is useful for:

        - validating the demo narrative;
        - exploring how horizon, slots, VICReg, and occlusion length affect confidence decay;
        - producing tables and plots before spending weeks on real pretraining.
        """
    )

    fig_idsw = px.line(
        demo["chart"],
        x="frame",
        y=["Baseline IDSW", "PredLatent IDSW"],
        title="Simulated cumulative ID switches",
        labels={"value": "ID switches", "variable": "Tracker"},
    )
    fig_idsw.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_idsw, use_container_width=True)

    st.markdown(
        """
        ### Architecture sketch

        ```text
        Frame
          |
          v
        Frozen DINOv2 patch features
          |
          v
        Slot Attention with temporal slot initialization
          |
          +--> slot latents z_t
          |
          v
        Per-slot transformer predictor + cross-slot attention
          |
          v
        Predicted latents z_hat_{t+1:t+H}
          |
          v
        Association cost:
          cosine(z_hat, z_obs) + Mahalanobis(box_hat, box_obs)
          |
          v
        Hungarian matching + learned lost threshold
        ```
        """
    )

with tab_bench:
    st.subheader("Simulated benchmark summary")
    st.caption(
        "These are synthetic proxy metrics generated from the current control settings. "
        "They are not TrackEval results on DanceTrack, SportsMOT, or DAVIS."
    )

    bench = build_benchmark(
        seed=seed,
        horizon=horizon,
        slots=slots,
        enable_predictor=enable_predictor,
        enable_vicreg=enable_vicreg,
    )

    st.dataframe(bench, use_container_width=True, hide_index=True)

    fig_hota = px.bar(
        bench,
        x="Method",
        y="HOTA mean",
        color="Occlusion bucket",
        barmode="group",
        error_y="HOTA std",
        title="Simulated HOTA by occlusion bucket",
    )
    fig_hota.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_hota, use_container_width=True)

    fig_idsw = px.bar(
        bench,
        x="Method",
        y="IDSW / 100 tracks mean",
        color="Occlusion bucket",
        barmode="group",
        error_y="IDSW / 100 tracks std",
        title="Simulated ID switches per 100 tracks",
    )
    fig_idsw.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_idsw, use_container_width=True)

with tab_ablation:
    st.subheader("Simulated ablations")
    st.caption(
        "Ablation values are produced by the same synthetic scoring model. "
        "They illustrate expected trends, not measured real-model performance."
    )

    abl = build_ablation(seed=seed)
    st.dataframe(abl, use_container_width=True, hide_index=True)

    fig_abl = px.bar(
        abl,
        x="Variant",
        y="IDSW / 100 mean",
        error_y="IDSW / 100 std",
        title="Simulated ID switches by ablation",
    )
    fig_abl.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_abl, use_container_width=True)

    fig_abl2 = px.scatter(
        abl,
        x="Slot variance",
        y="Effective rank",
        text="Variant",
        title="Collapse diagnostics: slot variance vs effective rank",
    )
    fig_abl2.update_traces(textposition="top center")
    fig_abl2.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_abl2, use_container_width=True)

with tab_fail:
    st.subheader("Failure gallery")

    failures = build_failures()
    st.dataframe(failures, use_container_width=True, hide_index=True)

    selected = st.selectbox("Inspect failure case", failures["Case"].tolist())
    row = failures[failures["Case"] == selected].iloc[0]

    st.markdown(f"### {row['Case']}: {row['Scenario']}")
    st.markdown(f"**Observed failure:** {row['Observed failure']}")
    st.markdown(f"**Root cause:** {row['Root cause']}")
    st.markdown(f"**Mitigation:** {row['Mitigation']}")

    xs = np.arange(0, 90)
    ys = np.exp(-xs / float(row["Confidence tau"]))
    decay_df = pd.DataFrame({"frame": xs, "confidence": ys}).set_index("frame")

    st.markdown("#### Confidence decay proxy")
    st.line_chart(decay_df)

with tab_model:
    st.subheader("Model card")

    st.markdown(
        """
        ### Name

        PredLatent-Track — interactive synthetic proof-of-concept.

        ### Task

        Multi-object tracking under occlusion, framed as label-free slot-latent prediction.

        ### Status

        This repository artifact is a simulator and demo interface. It is not a trained
        production tracker. A real implementation would require:

        - frozen DINOv2 ViT-S/14 feature extraction;
        - Slot Attention / SAVi-style temporal slot initialization;
        - EMA target encoder;
        - per-slot transformer predictor with cross-slot attention;
        - VICReg-style variance and covariance regularization;
        - pretraining on unlabeled YouTube-VOS / DAVIS / SportsMOT train videos;
        - evaluation with TrackEval on DanceTrack val and SportsMOT val.

        ### Intended use

        - research communication;
        - hypothesis visualization;
        - ablation planning;
        - demo scripting before expensive pretraining.

        ### Limitations

        - all tracking outputs are synthetic;
        - metrics are proxy metrics, not benchmark metrics;
        - no real appearance embeddings are computed;
        - no true Hungarian matching on real detections is performed;
        - no ONNX export or laptop-GPU FPS measurement is included in this two-file demo.

        ### Ethical / licensing note

        For a real deployment:

        - use only datasets and models under licenses permitting your intended use;
        - obtain consent for any person filmed in live demos;
        - clearly distinguish simulated, replayed, and recorded content;
        - report measured numbers honestly, including failures.
        """
    )