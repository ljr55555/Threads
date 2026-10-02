"""Split an uploaded image into flat colour layers ready for digitizing."""
import numpy as np
import cv2
from PIL import Image

MAX_WORK_PX = 1400          # working raster; bigger only costs time
MAX_COLORS = 32


def load(path):
    im = Image.open(path).convert('RGBA')
    if max(im.size) > MAX_WORK_PX:
        k = MAX_WORK_PX / max(im.size)
        im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))),
                       Image.LANCZOS)
    return np.array(im).astype(np.int16)


def foreground(rgba, bg_tol=26):
    """Alpha if present, otherwise flood the background colour from the border."""
    a = rgba[:, :, 3]
    if (a < 250).mean() > 0.02:
        return a > 128
    rgb = rgba[:, :, :3]
    h, w = rgb.shape[:2]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    vals, counts = np.unique(border.reshape(-1, 3), axis=0, return_counts=True)
    bg = vals[counts.argmax()]
    d = np.linalg.norm(rgb - bg, axis=2)
    fg = d > bg_tol
    # keep only regions connected to the interior, so speckle in the
    # background doesn't become a phantom colour layer
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((3, 3), np.uint8)).astype(bool)
    return fg


def quantize(rgba, n_colors, fg=None):
    """K-means the foreground into n flat colours. Returns layers ordered
    light -> dark, so darker outline colours sew last and cover the seams."""
    if fg is None:
        fg = foreground(rgba)
    rgb = rgba[:, :, :3].astype(np.float32)
    pix = rgb[fg]
    if len(pix) < 16:
        raise ValueError('the image looks empty once the background is removed')
    n_colors = int(max(1, min(n_colors, MAX_COLORS, len(np.unique(pix, axis=0)))))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, lbl, centers = cv2.kmeans(pix, n_colors, None, crit, 4,
                                 cv2.KMEANS_PP_CENTERS)
    lbl = lbl.ravel()
    layers = []
    idx = np.argwhere(fg)
    for c in range(n_colors):
        m = np.zeros(fg.shape, np.uint8)
        sel = idx[lbl == c]
        if len(sel) == 0:
            continue
        m[sel[:, 0], sel[:, 1]] = 255
        col = centers[c].round().astype(int).clip(0, 255)
        lum = 0.2126 * col[0] + 0.7152 * col[1] + 0.0722 * col[2]
        layers.append({
            'mask': m,
            'rgb': tuple(int(v) for v in col),
            'hex': '#%02X%02X%02X' % tuple(int(v) for v in col),
            'hex_int': (int(col[0]) << 16) | (int(col[1]) << 8) | int(col[2]),
            'lum': float(lum),
            'area_px': int((m > 0).sum()),
        })
    layers.sort(key=lambda L: -L['lum'])
    for i, L in enumerate(layers):
        L['name'] = 'Colour %d' % (i + 1)
        L['order'] = i + 1
    return layers


def stroke_stats(mask, scale_mm_per_px):
    """Widest and typical stroke width in mm — decides satin vs fill."""
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    d = dt[dt > 0]
    if d.size == 0:
        return 0.0, 0.0
    return float(2 * d.max() * scale_mm_per_px), float(2 * np.median(d) * scale_mm_per_px)
