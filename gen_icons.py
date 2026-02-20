"""Generate PWA icons for Domino Pernambucano."""
from PIL import Image, ImageDraw
import math

def draw_icon(size):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background rounded rect
    r = int(size * 0.15)
    draw.rounded_rectangle([0, 0, size-1, size-1], radius=r, fill='#0f3d1e')

    # Inner glow (simple lighter center)
    for i in range(20):
        alpha = int(30 - i * 1.5)
        if alpha <= 0:
            break
        spread = int(size * 0.3 * (1 - i/20))
        cx, cy = size//2, size//2
        draw.ellipse(
            [cx-spread, cy-spread, cx+spread, cy+spread],
            fill=(26, 92, 46, alpha)
        )

    # Domino tile dimensions
    tw = int(size * 0.38)
    th = int(size * 0.70)
    tx = (size - tw) // 2
    ty = (size - th) // 2
    tr = int(size * 0.03)

    # Tile shadow
    for s in range(6, 0, -1):
        a = int(40 - s * 6)
        draw.rounded_rectangle(
            [tx+s, ty+s+2, tx+tw+s, ty+th+s+2],
            radius=tr, fill=(0, 0, 0, max(a, 0))
        )

    # Tile body
    draw.rounded_rectangle(
        [tx, ty, tx+tw, ty+th],
        radius=tr, fill='#f0ead8', outline='#b8a88a', width=max(1, size//120)
    )

    # Divider line
    mid_y = ty + th // 2
    draw.line(
        [(tx + int(tw*0.1), mid_y), (tx + tw - int(tw*0.1), mid_y)],
        fill='#c8b898', width=max(1, size//200)
    )

    # Dots
    dot_r = max(2, int(size * 0.022))
    pad_x = int(tw * 0.27)
    top_cy = ty + th // 4
    bot_cy = ty + 3 * th // 4
    spread_y = int(th * 0.11)

    def dot(cx, cy):
        draw.ellipse(
            [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
            fill='#1a1a1a'
        )

    # Top: 6 (3 left, 3 right)
    for off in [-spread_y, 0, spread_y]:
        dot(tx + pad_x, top_cy + off)
        dot(tx + tw - pad_x, top_cy + off)

    # Bottom: 4 (2 left, 2 right)
    for off in [-spread_y, spread_y]:
        dot(tx + pad_x, bot_cy + off)
        dot(tx + tw - pad_x, bot_cy + off)

    return img

for sz in [192, 512]:
    img = draw_icon(sz)
    path = f'icon-{sz}.png'
    img.save(path)
    print(f'Saved {path}')
