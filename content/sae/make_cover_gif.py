#!/usr/bin/env python3
"""Generate sae_primer_cover.gif — the animated banner for 01_saes_from_scratch.

Run from the repo root:  .venv/bin/python content/sae/make_cover_gif.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.animation import FuncAnimation, PillowWriter

BG, FG, MUT = "#ffffff", "#1f2430", "#6b7280"
BLUE, ORANGE = "#4c8dd6", "#f08c00"
CELL_E, DENSE = "#c3c9d4", "#e4e8ef"
BLK_F, BLK_E = "#f2f4f8", "#c3c9d4"

FPS, NF = 25, 162
W, H = 1200, 675
SY = 585
TAPX = 470

H_X, H_W, H_H, H_TOPS = 110, 100, 50, [300, 356]
Z_X, Z_W, Z_H, Z_TOPS = 505, 190, 50, [230, 290, 350, 410]
HH_X = 990
NAMES = ["cat", "dog", "French", "CAPS"]
Z_DENSE = [0.9, 0.5, 0.8, 0.4]
Z_FINAL = [1.0, 0.0, 1.0, 0.0]
ACTIVE = (0, 2)
H_VAL = [1.7, 0.7]
HH0 = [1.2, 1.1]

def ease(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)

def phase(f, a, b):
    return float(ease((f - a) / max(b - a, 1)))

fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=150)   # 1800x1013 output
fig.patch.set_facecolor(BG)
ax.set_position([0, 0, 1, 1])          # axes fill the figure: data coords == pixels, true centering

def cell(x, y, w, h, fc, ec, lw=1.4, alpha=1.0):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1,rounding_size=6",
                                   fc=fc, ec=ec, lw=lw, alpha=alpha))

def dashed(x0, y0, x1, y1, t):
    n = max(2, int(14 * t))
    xs = np.linspace(x0, x1, 14)[:n]; ys = np.linspace(y0, y1, 14)[:n]
    for j in range(0, n - 1, 2):
        ax.plot(xs[j:j + 2], ys[j:j + 2], color=MUT, lw=1.3)

def draw(f):
    ax.clear(); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_facecolor(BG)

    # titles + status line (single, fixed spot)
    ax.text(W / 2, 42, "Sparse autoencoders from scratch", ha="center", va="center",
            color=FG, fontsize=27, fontweight="bold")
    ax.text(W / 2, 84, "how an SAE un-mixes a language model's residual stream",
            ha="center", va="center", color=MUT, fontsize=14)

    p_tok    = phase(f, 0, 20)
    p_hfill  = phase(f, 20, 30)
    p_up     = phase(f, 30, 40)
    p_dense  = phase(f, 40, 52)
    p_down   = phase(f, 52, 62)
    p_hh0    = phase(f, 62, 72)
    p_train  = phase(f, 76, 108)
    p_dash   = phase(f, 106, 120)
    p_check  = phase(f, 118, 130)
    p_label  = phase(f, 132, 148)

    if 0 < p_train < 1:
        ax.text(W / 2, 126, "training:  $z$ and $\\hat{h}$ improve together …", ha="center",
                color=MUT, fontsize=13.5, style="italic")
    elif p_check > 0:
        ax.text(W / 2, 126, r"trained so $\hat{h} \approx h$  ✓ — now name the latents that fire",
                ha="center", color=ORANGE, fontsize=13.5, style="italic", alpha=p_check)

    # loss + live error, top right stack
    ax.text(1170, 168, r"$\mathcal{L} = \|h-\hat{h}\|_2^2 + \lambda\|z\|_1$",
            ha="right", color=MUT, fontsize=13)
    hh = [HH0[i] + (H_VAL[i] - HH0[i]) * p_train for i in range(2)]
    if p_hh0 >= 1:
        err = float(np.hypot(H_VAL[0] - hh[0], H_VAL[1] - hh[1]))
        done = err < 0.005
        ax.text(1170, 200, r"error $\|h-\hat{h}\|_2 = $" + f"{err:.2f}" + ("  ✓" if done else ""),
                ha="right", color=(ORANGE if done else MUT), fontsize=11.5,
                fontweight="bold" if done else "normal")

    # ---- residual stream ----
    for a, b in [(158, 215), (305, 320), (410, 620), (740, 1170)]:
        ax.plot([a, b], [SY, SY], color=MUT, lw=1.5, alpha=0.6, solid_capstyle="round")
    ax.annotate("", xy=(1172, SY), xytext=(1155, SY),
                arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.5))
    for x, nm, bw in [(215, "ℓ−1", 90), (320, "ℓ", 90), (620, "block ℓ+1", 120)]:
        ax.add_patch(mp.FancyBboxPatch((x, SY - 29), bw, 58,
                     boxstyle="round,pad=2,rounding_size=9", fc=BLK_F, ec=BLK_E, lw=1.2))
        ax.text(x + bw / 2, SY + 1, nm, ha="center", va="center", color=MUT, fontsize=11.5)
    ax.text(955, SY + 44, "residual stream (frozen network)", ha="center",
            color=MUT, fontsize=11, style="italic")

    # static token chip at far left; travelling dot carries the activation
    ax.add_patch(mp.FancyBboxPatch((24, SY - 21), 132, 42,
                 boxstyle="round,pad=2,rounding_size=10", fc=BLUE, ec="none",
                 alpha=0.95, zorder=7))
    ax.text(90, SY + 1, "« le chat »", ha="center", va="center", color="white",
            fontsize=12.5, zorder=8)
    if 0 < p_tok < 1:
        ax.add_patch(mp.Circle((162 + (TAPX - 162) * p_tok, SY), 8, fc=BLUE, ec="none", zorder=8))
    ax.add_patch(mp.Circle((TAPX, SY), 7, fc=BLUE if p_tok >= 1 else BLK_E, ec="none", zorder=6))

    # elbow from tap to the h column
    ax.plot([TAPX - 6, TAPX - 6], [SY - 12, 500], color=MUT, lw=1.5, alpha=0.8)
    ax.plot([TAPX - 6, 160], [500, 500], color=MUT, lw=1.5, alpha=0.8)
    ax.annotate("", xy=(160, 414), xytext=(160, 502),
                arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.5))

    # ---- funnels with the equations inside ----
    ax.add_patch(mp.Polygon([(H_X + H_W + 2, 300), (H_X + H_W + 2, 406),
                             (Z_X - 2, 460), (Z_X - 2, 230)],
                            closed=True, fc="#8a93a3", alpha=0.07, ec="#8a93a3", lw=0.8))
    ax.add_patch(mp.Polygon([(Z_X + Z_W + 2, 230), (Z_X + Z_W + 2, 460),
                             (HH_X - 2, 406), (HH_X - 2, 300)],
                            closed=True, fc="#8a93a3", alpha=0.07, ec="#8a93a3", lw=0.8))
    ax.text(358, 330, "encoder", ha="center", color=MUT, fontsize=11.5, style="italic")
    ax.text(358, 364, r"$z = $ReLU$(W_{\rm enc}\,h + b_{\rm enc})$", ha="center",
            color=FG, fontsize=13)
    ax.text(842, 330, "decoder", ha="center", color=MUT, fontsize=11.5, style="italic")
    ax.text(842, 364, r"$\hat{h} = W_{\rm dec}\,z + b_{\rm dec}$", ha="center",
            color=FG, fontsize=13)

    # ---- h cells ----
    for i, y in enumerate(H_TOPS):
        on = p_hfill > i / 2
        cell(H_X, y, H_W, H_H, BLUE if on else BG, BLUE if on else CELL_E)
        if on:
            v = H_VAL[i] * phase(f, 20 + 5 * i, 30)
            ax.text(H_X + H_W / 2, y + H_H / 2, f"{v:.1f}", ha="center", va="center",
                    color="white", fontsize=17, fontweight="bold")
    ax.text(H_X + H_W / 2, 272, r"$h \in \mathbb{R}^m$", ha="center", color=FG, fontsize=15)

    if 0 < p_up < 1:
        ax.add_patch(mp.Circle((H_X + H_W + 6 + (Z_X - H_X - H_W - 12) * p_up,
                                390 - 60 * p_up), 8, fc=BLUE, ec="none", zorder=6))

    # ---- z cells (anonymous until the end) ----
    for i, y in enumerate(Z_TOPS):
        if i in ACTIVE:
            v = Z_DENSE[i] + (Z_FINAL[i] - Z_DENSE[i]) * p_train
            if p_train > 0:
                cell(Z_X, y, Z_W, Z_H, ORANGE if p_train > 0.5 else DENSE, "none",
                     alpha=0.25 + 0.75 * max(p_train, 0.3))
            elif p_dense > 0:
                cell(Z_X, y, Z_W, Z_H, DENSE, "none", alpha=p_dense)
        else:
            v = Z_DENSE[i] * (1 - p_train)
            a_fill = p_dense * (1 - p_train)
            if a_fill > 0.02:
                cell(Z_X, y, Z_W, Z_H, DENSE, "none", alpha=a_fill)
        cell(Z_X, y, Z_W, Z_H, "none", CELL_E)
        if p_dense > 0.3:
            val_c = ("white" if (i in ACTIVE and p_train > 0.5) else FG)
            vv = v if (i in ACTIVE or p_train < 0.98) else 0.0
            ax.text(Z_X + Z_W - 18, y + Z_H / 2, f"{vv:.1f}", ha="right", va="center",
                    color=val_c, fontsize=15, fontweight="bold")
        if i in ACTIVE and p_label > 0:
            ax.text(Z_X + 18, y + Z_H / 2, NAMES[i], ha="left", va="center",
                    color="white", fontsize=13.5, alpha=p_label, fontweight="bold")
    ax.text(Z_X + Z_W / 2, 202, r"$z \in \mathbb{R}^M$", ha="center", color=FG, fontsize=15)

    if 0 < p_down < 1:
        ax.add_patch(mp.Circle((Z_X + Z_W + 6 + (HH_X - Z_X - Z_W - 12) * p_down,
                                330 + 55 * p_down), 8, fc=ORANGE, ec="none", zorder=6))

    # ---- h-hat cells ----
    for i, y in enumerate(H_TOPS):
        on = p_hh0 > i / 2
        cell(HH_X, y, H_W, H_H, BLUE if on else BG, BLUE if on else CELL_E,
             alpha=0.5 if on else 1.0)
        if on:
            ax.text(HH_X + H_W / 2, y + H_H / 2, f"{hh[i]:.1f}", ha="center", va="center",
                    color=FG, fontsize=17, fontweight="bold")
    ax.text(HH_X + H_W / 2, 272, r"$\hat{h} \in \mathbb{R}^m$", ha="center", color=FG, fontsize=15)

    # dashed comparison back to the tap (squared-off route above the blocks)
    if p_dash > 0:
        cx = HH_X + H_W / 2
        t1 = min(1.0, 3 * p_dash)
        dashed(cx, 414, cx, 414 + 121 * t1, t1)
        t2 = np.clip(3 * p_dash - 1, 0, 1)
        if t2 > 0:
            dashed(cx, 535, cx + (TAPX + 6 - cx) * t2, 535, t2)
        t3 = np.clip(3 * p_dash - 2, 0, 1)
        if t3 > 0:
            dashed(TAPX + 6, 535, TAPX + 6, 535 + (SY - 549), t3)

anim = FuncAnimation(fig, draw, frames=NF, interval=1000 / FPS)
out = "content/sae/sae_primer_cover.gif"
anim.save(out, writer=PillowWriter(fps=FPS))
print("saved", out)
