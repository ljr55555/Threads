"""Print worksheet PDF, ported from Ink/Stitch's print templates.

Ink/Stitch renders its worksheet as HTML in a browser and the user prints it
to PDF; here the same layout is drawn straight to PDF with reportlab:

* page 1 - client/overview page (print_overview.html): design preview, job
  details, colour palette with matched thread names, and a quote block ported
  from embTools' quote sheet (quotesheet.cpp: digitizing = price-per-1000 x
  stitches; total = setup + marked-up product + digitizing - discount);
* page 2 - operator detailed view (operator_detailedview.html): one row per
  colour block with swatch, thread, stitch count, estimated time, stops and
  trims, and a notes line.

That is the 'classic' layout. The default 'production' layout is the
one-page production worksheet digitizing studios print: a boxed header
with the design name and a stitches/size box, the sewn preview on the left,
machine statistics (extents from the origin, area, stitch/jump lengths,
thread and bobbin usage) and the stop sequence (needle, colour, stitches,
thread code, name, chart) on the right, and an authors/dates footer bar.
The quote sheet follows on its own page when quote parameters are given.
"""
import datetime
import math
import os

import pystitch
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfgen.canvas import Canvas

PAGE_W, PAGE_H = A4
MARGIN = 15 * mm
INK = HexColor('#12161c')
MUTED = HexColor('#6b7480')
LINE = HexColor('#c9ced4')

# appearance themes (Design panel -> Worksheet): reportlab's built-in faces
FONTS = {'Helvetica': ('Helvetica', 'Helvetica-Bold'),
         'Times': ('Times-Roman', 'Times-Bold'),
         'Courier': ('Courier', 'Courier-Bold')}


def _apply_theme(c, theme, logo_path):
    t = theme or {}
    c._body, c._bold = FONTS.get(t.get('font', 'Helvetica'), FONTS['Helvetica'])
    c._accent = HexColor(t.get('accent', '#12161C'))
    c._theme = t
    c._logo = logo_path if (logo_path and t.get('show_logo', True)) else None


def block_stats(pattern):
    """Per colour block: stitches, trims, stops (jumps left untrimmed)."""
    blocks = [{'stitches': 0, 'trims': 0, 'stops': 0}]
    prev_cmd = None
    for x, y, cmd in pattern.stitches:
        c = cmd & 0xFF
        if c == pystitch.COLOR_CHANGE:
            blocks.append({'stitches': 0, 'trims': 0, 'stops': 0})
        elif c == pystitch.STITCH:
            blocks[-1]['stitches'] += 1
        elif c == pystitch.TRIM:
            blocks[-1]['trims'] += 1
        elif c == pystitch.JUMP and prev_cmd != pystitch.JUMP:
            blocks[-1]['stops'] += 1
        prev_cmd = c
    return blocks


def _est_time(stitches, spm=700):
    minutes = stitches / spm
    return '%d:%02d min' % (int(minutes), int(round(minutes % 1 * 60)))


def _text(c, x, y, s, size=9, bold=False, color=INK, align='left'):
    c.setFillColor(color)
    font = getattr(c, '_bold', 'Helvetica-Bold') if bold else getattr(c, '_body', 'Helvetica')
    c.setFont(font, size)
    if align == 'right':
        c.drawRightString(x, y, str(s))
    elif align == 'center':
        c.drawCentredString(x, y, str(s))
    else:
        c.drawString(x, y, str(s))


def _header(c, title, subtitle):
    from reportlab.lib.utils import ImageReader
    t = getattr(c, '_theme', {}) or {}
    x_title = MARGIN
    date_x = PAGE_W - MARGIN
    if getattr(c, '_logo', None) and os.path.exists(c._logo):
        try:
            img = ImageReader(c._logo)
            iw, ih = img.getSize()
            lh = float(t.get('logo_h_mm', 12)) * mm
            lw = lh * iw / max(ih, 1)
            if t.get('logo_pos', 'right') == 'left':
                c.drawImage(img, MARGIN, PAGE_H - MARGIN - lh + 11, lw, lh,
                            preserveAspectRatio=True, anchor='nw', mask='auto')
                x_title = MARGIN + lw + 5 * mm
            else:
                c.drawImage(img, PAGE_W - MARGIN - lw, PAGE_H - MARGIN - lh + 11,
                            lw, lh, preserveAspectRatio=True, anchor='nw', mask='auto')
                date_x = PAGE_W - MARGIN - lw - 4 * mm
        except Exception:
            pass
    _text(c, x_title, PAGE_H - MARGIN, title, 16, bold=True)
    _text(c, x_title, PAGE_H - MARGIN - 14, subtitle, 9, color=MUTED)
    _text(c, date_x, PAGE_H - MARGIN, datetime.date.today().isoformat(),
          9, color=MUTED, align='right')
    c.setStrokeColor(getattr(c, '_accent', INK))
    c.setLineWidth(1.1)
    c.line(MARGIN, PAGE_H - MARGIN - 22, PAGE_W - MARGIN, PAGE_H - MARGIN - 22)


def _footer(c):
    c.setStrokeColor(LINE)
    c.line(MARGIN, MARGIN, PAGE_W - MARGIN, MARGIN)
    t = getattr(c, '_theme', {}) or {}
    _text(c, MARGIN, MARGIN - 10,
          t.get('footer') or
          'StitchForge worksheet — layout after the Ink/Stitch print worksheet (inkstitch.org)',
          7, color=MUTED)
    _text(c, PAGE_W - MARGIN, MARGIN - 10, 'page %d' % c.getPageNumber(),
          7, color=MUTED, align='right')


def _kv_rows(c, x, y, rows, key_w=42 * mm, lh=13):
    for k, v in rows:
        _text(c, x, y, k, 9, color=MUTED)
        _text(c, x + key_w, y, v, 9)
        y -= lh
    return y


QUOTE_FIELDS = ('setup', 'price_per_1000', 'min_digitizing', 'garment_qty', 'garment_base',
                'markup_pct', 'run_per_1000', 'colour_fee', 'extra_per_piece',
                'discount_pct', 'rush_pct', 'tax_pct')


def quote(stitches, setup=0.0, price_per_1000=0.0, garment_qty=0, garment_base=0.0,
          markup_pct=0.0, discount_pct=0.0, min_digitizing=0.0, run_per_1000=0.0,
          colour_changes=0, colour_fee=0.0, extra_per_piece=0.0, rush_pct=0.0, tax_pct=0.0,
          **_ignored):
    """The quote calculation, grown from embTools' quote sheet (quotesheet.cpp).

    One-time: setup fee + digitizing (price per 1000 stitches, at least the
    minimum). Per piece: the garment marked up, plus the embroidery run
    (run price per 1000 stitches + a fee per colour change + any extra per
    piece). Then the discount off the per-piece work, a rush surcharge,
    tax, the total and the price per piece."""
    stitches = float(stitches or 0)
    qty = int(garment_qty or 0)
    digitizing = 0.0
    if price_per_1000 or min_digitizing:
        digitizing = max(float(min_digitizing or 0), price_per_1000 / 1000.0 * stitches)
    product = qty * garment_base
    marked_up = product * (1 + markup_pct / 100.0)
    run_piece = run_per_1000 / 1000.0 * stitches + colour_fee * int(colour_changes or 0) + extra_per_piece
    run = qty * run_piece
    discount = discount_pct / 100.0 * (marked_up + run)
    rush = rush_pct / 100.0 * (marked_up + run + digitizing - discount)
    subtotal = setup + digitizing + marked_up + run - discount + rush
    tax = tax_pct / 100.0 * subtotal
    total = subtotal + tax
    return {'product': product, 'marked_up': marked_up, 'discount': discount,
            'digitizing': digitizing, 'setup': setup, 'run': run, 'run_piece': run_piece,
            'rush': rush, 'subtotal': subtotal, 'tax': tax, 'total': total,
            'per_piece': total / qty if qty else 0.0, 'qty': qty}


def quote_rows(stitches, quote_params):
    """(label, amount) lines for a quote, only the ones that apply."""
    q = quote(stitches, **quote_params)
    g = lambda k, d=0: quote_params.get(k, d) or d
    rows = []
    if q['setup']:
        rows.append(('Setup fee', '$%.2f' % q['setup']))
    if q['digitizing']:
        lab = 'Digitizing (%s st @ $%.2f/1000' % ('{:,}'.format(int(stitches)), g('price_per_1000'))
        if g('min_digitizing') and q['digitizing'] <= g('min_digitizing'):
            lab += ', minimum'
        rows.append((lab + ')', '$%.2f' % q['digitizing']))
    if q['marked_up']:
        rows.append(('Garments (%d × $%.2f%s)' % (q['qty'], g('garment_base'),
                     ', %+.0f%% markup' % g('markup_pct') if g('markup_pct') else ''), '$%.2f' % q['marked_up']))
    if q['run']:
        rows.append(('Embroidery (%d × $%.2f per piece)' % (q['qty'], q['run_piece']), '$%.2f' % q['run']))
    if q['discount']:
        rows.append(('Discount (%.0f%%)' % g('discount_pct'), '-$%.2f' % q['discount']))
    if q['rush']:
        rows.append(('Rush (%.0f%%)' % g('rush_pct'), '$%.2f' % q['rush']))
    if q['tax']:
        rows.append(('Subtotal', '$%.2f' % q['subtotal']))
        rows.append(('Tax (%.2f%%)' % g('tax_pct'), '$%.2f' % q['tax']))
    return q, rows


def build_quote(path, stitches, quote_params, design_name='design', client='',
                theme=None, logo_path=None, colour_changes=0, notes=''):
    """A one-page quote PDF on its own."""
    c = Canvas(path, pagesize=letter)
    c.setTitle('%s — quote' % design_name)
    _apply_theme(c, theme, logo_path)
    _quote_page(c, stitches, dict(quote_params, colour_changes=colour_changes), design_name,
                client=client, notes=notes)
    c.showPage()
    c.save()
    return path


# ------------------------------------------------ production worksheet
APP_LINE = 'Threads Studio - Designing'
MACHINE_FORMAT = 'Tajima'           # the native file the app saves is DST
TAKE_UP_MM = 1.6                    # top thread used per penetration
BOBBIN_RATIO = 1 / 3.0              # bobbin shows ~1/3 of the column width


def production_stats(pattern):
    """Machine-sheet numbers, all from the stitch list (0.1 mm units)."""
    st = pattern.stitches
    xs = [x for x, y, c in st if (c & 0xFF) in (pystitch.STITCH, pystitch.JUMP)]
    ys = [y for x, y, c in st if (c & 0xFF) in (pystitch.STITCH, pystitch.JUMP)]
    if not xs:
        xs = ys = [0.0]
    lens, jumps = [], []
    trims = 0
    prev = None
    run_jump = 0.0
    for x, y, c in st:
        k = c & 0xFF
        if k == pystitch.TRIM:
            trims += 1
        if prev is not None and k in (pystitch.STITCH, pystitch.JUMP):
            d = math.hypot(x - prev[0], y - prev[1]) / 10.0
            if k == pystitch.STITCH:
                if d > 0:
                    lens.append(d)
                if run_jump:
                    jumps.append(run_jump)
                run_jump = 0.0
            else:
                run_jump += d
        if k in (pystitch.STITCH, pystitch.JUMP):
            prev = (x, y)
    if run_jump:
        jumps.append(run_jump)
    end = next(((x, y) for x, y, c in reversed(st)
                if (c & 0xFF) in (pystitch.STITCH, pystitch.JUMP)), (0.0, 0.0))
    w_mm = (max(xs) - min(xs)) / 10.0
    h_mm = (max(ys) - min(ys)) / 10.0
    top_mm = sum(lens) + len(lens) * TAKE_UP_MM
    return {
        'left_mm': -min(xs) / 10.0, 'right_mm': max(xs) / 10.0,
        'up_mm': -min(ys) / 10.0, 'down_mm': max(ys) / 10.0,
        'width_mm': w_mm, 'height_mm': h_mm,
        'end_x_in': end[0] / 254.0, 'end_y_in': -end[1] / 254.0,
        'area_in2': (w_mm / 25.4) * (h_mm / 25.4),
        'max_stitch_mm': max(lens) if lens else 0.0,
        'min_stitch_mm': min(lens) if lens else 0.0,
        'max_jump_mm': max(jumps) if jumps else 0.0,
        'thread_ft': top_mm / 304.8,
        'bobbin_ft': top_mm * BOBBIN_RATIO / 304.8,
        'stitches': len(lens) + 1 if lens else 0,
        'trims': trims,
    }


def _fit(c, s, width, font, size):
    """Truncate s with an ellipsis so it fits `width` points."""
    s = str(s)
    if c.stringWidth(s, font, size) <= width:
        return s
    while s and c.stringWidth(s + '…', font, size) > width:
        s = s[:-1]
    return s + '…'


def _stamp(t):
    return '%d/%d/%d %d:%02d:%02d %s' % (
        t.month, t.day, t.year, (t.hour % 12) or 12, t.minute, t.second,
        'AM' if t.hour < 12 else 'PM')


def _production_page(c, pattern, report, layers, thread_matches, preview_png,
                     design_name, client, saved_at, title='', pages=1):
    from reportlab.lib.utils import ImageReader
    W, H = letter
    body, bold = c._body, c._bold
    theme = c._theme or {}
    m = 0.22 * 72                         # outer frame margin
    x0, x1, y0, y1 = m, W - m, m, H - m
    head_h = 88.0
    foot_h = 14.0
    stats_w = 98.0                         # header stats box (top right)
    col_x = x0 + (x1 - x0) * 0.672         # right column divider
    ps = production_stats(pattern)
    blocks = block_stats(pattern)

    c.setStrokeColor(black)
    c.setLineWidth(0.8)
    c.rect(x0, y0, x1 - x0, y1 - y0)
    c.line(x0, y1 - head_h, x1, y1 - head_h)
    c.line(x1 - stats_w, y1, x1 - stats_w, y1 - head_h)
    c.line(x0, y0 + foot_h, x1, y0 + foot_h)
    c.line(col_x, y1 - head_h, col_x, y0 + foot_h)

    # ---- header, left: title block
    tx = x0 + 3
    _text(c, tx, y1 - 13, 'Production Worksheet', 12.5, bold=True, color=c._accent)
    _text(c, tx, y1 - 26, theme.get('footer') or APP_LINE, 8)
    big = '*%s*' % design_name.upper()
    size = 20
    avail = (x1 - stats_w) - tx - 6
    while size > 10 and c.stringWidth(big, body, size) > avail:
        size -= 1
    _text(c, tx, y1 - 47, big, size)
    _text(c, tx, y1 - 66, 'Design:', 8)
    _text(c, tx + 34, y1 - 66, _fit(c, design_name, avail - 36, bold, 11), 11, bold=True)
    _text(c, tx, y1 - 82, 'Title:', 8)
    if title:
        _text(c, tx + 34, y1 - 82, _fit(c, title, avail - 36, body, 8), 8)
    if getattr(c, '_logo', None) and os.path.exists(c._logo):
        try:
            img = ImageReader(c._logo)
            iw, ih = img.getSize()
            lh = min(float(theme.get('logo_h_mm', 12)) * mm, head_h - 30)
            lw = lh * iw / max(ih, 1)
            lx = (x1 - stats_w - 6 - lw) if theme.get('logo_pos', 'right') != 'left' \
                else x0 + 0.42 * (x1 - stats_w - x0)
            c.drawImage(img, lx, y1 - 6 - lh, lw, lh, preserveAspectRatio=True,
                        mask='auto')
        except Exception:
            pass

    # ---- header, right: stitches / size box
    unique = []
    for L in layers:
        if L['hex'].upper() not in unique:
            unique.append(L['hex'].upper())
    sx = x1 - stats_w + 3
    rows = [('Stitches:', '{:,}'.format(report.get('stitches', ps['stitches']))),
            ('Height:', '%.2f in' % (ps['height_mm'] / 25.4)),
            ('Width:', '%.2f in' % (ps['width_mm'] / 25.4)),
            ('Colors:', str(len(unique))),
            ('Colorway:', 'Colorway 1'),
            ('Zoom:', '%ZOOM%')]
    zoom_y = None
    for i, (k, v) in enumerate(rows):
        yy = y1 - 10 - i * 12.2
        _text(c, sx, yy, k, 8)
        if v == '%ZOOM%':
            zoom_y = yy
        else:
            _text(c, sx + 47, yy, v, 8)

    # ---- preview, left area
    area_top = y1 - head_h - 6
    area_bot = y0 + foot_h + 6
    ax0, ax1 = x0 + 6, col_x - 6
    zoom = 0.0
    if preview_png and os.path.exists(preview_png):
        img = ImageReader(preview_png)
        iw, ih = img.getSize()
        box_w, box_h = ax1 - ax0, (area_top - area_bot) * 0.78
        sc = min(box_w / iw, box_h / ih)
        dw, dh = iw * sc, ih * sc
        c.drawImage(img, ax0 + (box_w - dw) / 2, area_top - 40 - dh - (box_h - dh) / 2,
                    dw, dh, mask=[250, 255, 250, 255, 250, 255])
        # the PNG has a 3% pad on each side of the design's long edge
        long_mm = max(ps['width_mm'], ps['height_mm'], 0.1)
        drawn_long = max(dw, dh) / 1.06
        zoom = drawn_long / (long_mm / 25.4 * 72)
    if zoom_y is not None:
        _text(c, sx + 47, zoom_y, '%.2f' % zoom, 8)

    # ---- right column: machine statistics
    rx = col_x + 3
    vx = rx + 72
    lh = 12.6
    y = y1 - head_h - 10
    stat_rows = [
        ('Machine format:', MACHINE_FORMAT),
        ('Color changes:', str(max(0, len(blocks) - 1))),
        ('Stops:', str(len(blocks))),
        ('Trims:', str(ps['trims'])),
        ('Appliqués:', '0'),
        ('Left:', '%.1f mm' % ps['left_mm']),
        ('Right:', '%.1f mm' % ps['right_mm']),
        ('Up:', '%.1f mm' % ps['up_mm']),
        ('Down:', '%.1f mm' % ps['down_mm']),
        ('EndX:', '%.2f in' % ps['end_x_in']),
        ('EndY:', '%.2f in' % ps['end_y_in']),
        ('Area', '%.2f in²' % ps['area_in2']),
        ('Max stitch:', '%.1f mm' % ps['max_stitch_mm']),
        ('Min stitch:', '%.1f mm' % ps['min_stitch_mm']),
        ('Max jump:', '%.1f mm' % ps['max_jump_mm']),
        ('Total thread:', '%.2fft' % ps['thread_ft']),
        ('Total bobbin:', '%.2fft' % ps['bobbin_ft']),
    ]
    for k, v in stat_rows:
        _text(c, rx, y, k, 8)
        _text(c, vx, y, v, 8)
        y -= lh
    c.line(col_x, y + lh - 3.5, x1, y + lh - 3.5)

    # ---- stop sequence
    _text(c, rx, y, 'Stop Sequence:', 8)
    y -= lh
    cols = {'#': rx, 'N#': rx + 12, 'Color': rx + 25, 'St.': rx + 79,
            'Code': rx + 82, 'Name': rx + 106, 'Chart': x1 - 3}
    for k, align in (('#', 'l'), ('N#', 'l'), ('Color', 'l'), ('St.', 'r'),
                     ('Code', 'l'), ('Name', 'l'), ('Chart', 'r')):
        _text(c, cols[k], y, k, 8, bold=True, align='right' if align == 'r' else 'left')
        w = c.stringWidth(k, bold, 8)
        ux = cols[k] - w if align == 'r' else cols[k]
        c.setLineWidth(0.5)
        c.line(ux, y - 1.5, ux + w, y - 1.5)
    y -= lh
    for i, b in enumerate(blocks):
        if y < y0 + foot_h + 10:
            break
        L = layers[i] if i < len(layers) else {'hex': '#888888', 'name': 'Colour %d' % (i + 1)}
        needle = unique.index(L['hex'].upper()) + 1 if L['hex'].upper() in unique else i + 1
        t = thread_matches[i] if thread_matches and i < len(thread_matches) else None
        code = (t.get('thread_number') or '') if t else ''
        name = (t.get('thread_name') if t else None) or L.get('name', '')
        chart = t.get('palette', 'Default') if t else 'Default'
        _text(c, cols['#'], y, '%d.' % (i + 1), 8)
        _text(c, cols['N#'], y, str(needle), 8)
        c.setFillColor(HexColor(L['hex']))
        c.setStrokeColor(black)
        c.setLineWidth(0.5)
        c.rect(cols['Color'], y - 2, 24, 9, fill=1)
        _text(c, cols['St.'], y, '{:,}'.format(b['stitches']), 7.5, align='right')
        _text(c, cols['Code'], y, _fit(c, code, 20, body, 7.5), 7.5)
        # chart = the thread brand; drop the line name when it won't fit
        fs = 7.5
        if c.stringWidth(chart, body, fs) > 40:
            chart = chart.split(' ')[0]
        chart = _fit(c, chart, 40, body, fs)
        name_w = cols['Chart'] - c.stringWidth(chart, body, fs) - cols['Name'] - 4
        _text(c, cols['Name'], y, _fit(c, name, name_w, body, fs), fs)
        _text(c, cols['Chart'], y, chart, fs, align='right')
        y -= lh
    next_block = min(len(blocks), i + 1) if blocks else 0
    c.setStrokeColor(black)
    c.setLineWidth(0.8)
    c.line(col_x, y + lh - 3.5, x1, y + lh - 3.5)

    # ---- footer bar
    fy = y0 + 4
    now = datetime.datetime.now()
    _text(c, x0 + 3, fy, 'Authors:%s' % (('  ' + client) if client else ''), 7.5)
    _text(c, x0 + 150, fy, 'Design last saved : %s' % _stamp(saved_at or now), 7.5)
    _text(c, x0 + 345, fy, 'Date printed: %s' % _stamp(now), 7.5)
    _text(c, x1 - 3, fy, 'Page %d of %d' % (c.getPageNumber(), pages), 7.5, align='right')
    return next_block


def _production_stop_page(c, blocks, layers, thread_matches, start, design_name,
                          client, saved_at, pages):
    """Draw a full-width continuation page for the production stop sequence."""
    W, H = letter
    body = c._body
    m = 0.22 * 72
    x0, x1, y0, y1 = m, W - m, m, H - m
    c.setStrokeColor(black)
    c.setLineWidth(0.8)
    c.rect(x0, y0, x1 - x0, y1 - y0)

    _text(c, x0 + 8, y1 - 20, 'Production Worksheet', 12.5, bold=True, color=c._accent)
    _text(c, x0 + 8, y1 - 35, design_name, 9)
    _text(c, x0 + 8, y1 - 55, 'Stop Sequence (continued)', 10, bold=True)
    c.line(x0, y1 - 64, x1, y1 - 64)

    cols = {'#': x0 + 8, 'N#': x0 + 35, 'Color': x0 + 66, 'St.': x0 + 151,
            'Code': x0 + 158, 'Name': x0 + 245, 'Chart': x1 - 8}
    y = y1 - 82
    for key, align in (('#', 'l'), ('N#', 'l'), ('Color', 'l'), ('St.', 'r'),
                       ('Code', 'l'), ('Name', 'l'), ('Chart', 'r')):
        _text(c, cols[key], y, key, 8, bold=True,
              align='right' if align == 'r' else 'left')
    c.line(x0 + 5, y - 4, x1 - 5, y - 4)
    y -= 16

    unique = []
    for layer in layers:
        if layer['hex'].upper() not in unique:
            unique.append(layer['hex'].upper())
    index = start
    while index < len(blocks) and y >= y0 + 30:
        block = blocks[index]
        layer = layers[index] if index < len(layers) else {
            'hex': '#888888', 'name': 'Colour %d' % (index + 1)}
        needle = unique.index(layer['hex'].upper()) + 1 \
            if layer['hex'].upper() in unique else index + 1
        match = thread_matches[index] if thread_matches and index < len(thread_matches) else None
        code = (match.get('thread_number') or '') if match else ''
        name = (match.get('thread_name') if match else None) or layer.get('name', '')
        chart = match.get('palette', 'Default') if match else 'Default'
        fs = 7.5
        if c.stringWidth(chart, body, fs) > 70:
            chart = chart.split(' ')[0]
        chart = _fit(c, chart, 70, body, fs)
        name_w = cols['Chart'] - c.stringWidth(chart, body, fs) - cols['Name'] - 8

        _text(c, cols['#'], y, '%d.' % (index + 1), 8)
        _text(c, cols['N#'], y, str(needle), 8)
        c.setFillColor(HexColor(layer['hex']))
        c.setStrokeColor(black)
        c.rect(cols['Color'], y - 2, 72, 9, fill=1)
        _text(c, cols['St.'], y, '{:,}'.format(block['stitches']), fs, align='right')
        _text(c, cols['Code'], y, _fit(c, code, 80, body, fs), fs)
        _text(c, cols['Name'], y, _fit(c, name, name_w, body, fs), fs)
        _text(c, cols['Chart'], y, chart, fs, align='right')
        y -= 14
        index += 1

    c.line(x0 + 5, y + 10, x1 - 5, y + 10)
    now = datetime.datetime.now()
    _text(c, x0 + 3, y0 + 4, 'Authors:%s' % (('  ' + client) if client else ''), 7.5)
    _text(c, x0 + 150, y0 + 4, 'Design last saved : %s' % _stamp(saved_at or now), 7.5)
    _text(c, x1 - 3, y0 + 4, 'Page %d of %d' % (c.getPageNumber(), pages),
          7.5, align='right')
    return index


def _quote_page(c, stitches, quote_params, design_name, client='', notes=''):
    """The quote on its own page."""
    import datetime
    W, H = c._pagesize
    q, rows = quote_rows(stitches, quote_params)
    x = 0.5 * 72
    y = H - 0.6 * 72
    _text(c, x, y, 'Quote — %s' % design_name, 14, bold=True, color=c._accent)
    _text(c, W - x, y, datetime.date.today().strftime('%b %d, %Y'), 9, align='right')
    if client:
        y -= 15
        _text(c, x, y, 'For: %s' % client, 10)
    y -= 10
    c.setStrokeColor(black)
    c.line(x, y, W - x, y)
    y -= 20
    for k, v in rows:
        _text(c, x, y, k, 10)
        _text(c, W - x, y, v, 10, align='right')
        y -= 16
    c.line(x, y + 10, W - x, y + 10)
    _text(c, x, y - 4, 'Total', 11, bold=True)
    _text(c, W - x, y - 4, '$%.2f' % q['total'], 11, bold=True, align='right')
    if q['qty']:
        y -= 16
        _text(c, x, y - 4, 'Per piece (%d)' % q['qty'], 10)
        _text(c, W - x, y - 4, '$%.2f' % q['per_piece'], 10, align='right')
    if notes:
        y -= 30
        for line in str(notes).splitlines()[:12]:
            _text(c, x, y, line[:110], 9)
            y -= 13


def build_production(path, pattern, report, layers, thread_matches=None,
                     preview_png=None, design_name='design', quote_params=None,
                     client='', theme=None, logo_path=None, saved_at=None, title=''):
    c = Canvas(path, pagesize=letter)
    c.setTitle('%s — production worksheet' % design_name)
    _apply_theme(c, theme, logo_path)
    blocks = block_stats(pattern)
    continuation_pages = int(math.ceil(max(0, len(blocks) - 32) / 48.0))
    pages = 1 + continuation_pages + (1 if quote_params else 0)
    next_block = _production_page(c, pattern, report, layers, thread_matches, preview_png,
                                  design_name, client, saved_at, title=title, pages=pages)
    c.showPage()
    while next_block < len(blocks):
        _apply_theme(c, theme, logo_path)
        next_block = _production_stop_page(c, blocks, layers, thread_matches, next_block,
                                           design_name, client, saved_at, pages)
        c.showPage()
    if quote_params:
        _apply_theme(c, theme, logo_path)
        _quote_page(c, report.get('stitches', 0), quote_params, design_name)
        c.showPage()
    c.save()
    return path


def build(path, pattern, report, layers, thread_matches=None, preview_png=None,
          design_name='design', quote_params=None, spm=700, client='',
          theme=None, logo_path=None, layout='production', saved_at=None,
          title=''):
    """Write the worksheet PDF to `path`.

    layout: 'production' (one-page production worksheet, the default) or
    'classic' (Ink/Stitch overview + operator pages).

    report: engine.qa_report dict (or the lighter lettering report).
    layers: [{'name','hex', ...}] in sew order.
    thread_matches: optional threads.match_layers output, same order.
    """
    if layout != 'classic':
        return build_production(path, pattern, report, layers, thread_matches,
                                preview_png, design_name, quote_params, client,
                                theme, logo_path, saved_at, title)
    c = Canvas(path, pagesize=A4)
    c.setTitle('%s — embroidery worksheet' % design_name)
    _apply_theme(c, theme, logo_path)

    # ---------------------------------------------------- page 1: overview
    _header(c, design_name, 'Embroidery worksheet — client overview')

    y_top = PAGE_H - MARGIN - 36
    # preview, left column
    img_w = 95 * mm
    img_h = 95 * mm
    if preview_png and os.path.exists(preview_png):
        from reportlab.lib.utils import ImageReader
        img = ImageReader(preview_png)
        iw, ih = img.getSize()
        s = min(img_w / iw, img_h / ih)
        c.drawImage(img, MARGIN, y_top - ih * s, iw * s, ih * s,
                    preserveAspectRatio=True, anchor='nw')
        img_h = ih * s

    # job details, right column
    x2 = MARGIN + 103 * mm
    stitches = report.get('stitches', 0)
    rows = [
        ('Design box size', '%.1f × %.1f mm  (%.2f × %.2f in)' % (
            report.get('width_mm', 0), report.get('height_mm', 0),
            report.get('width_mm', 0) / 25.4, report.get('height_mm', 0) / 25.4)),
        ('Total stitch count', '{:,}'.format(stitches)),
        ('Unique colours', str(len(layers))),
        ('Colour blocks', str(max(1, report.get('colour_changes', 0) + 1))),
        ('Total stops / trims', '%s / %s' % (report.get('travels', '—'), report.get('trimmed', '—'))),
        ('Estimated time @%d spm' % spm, _est_time(stitches, spm)),
    ]
    if report.get('hoop_clearance_mm') is not None:
        rows.append(('Hoop clearance', '%.1f mm' % report['hoop_clearance_mm']))
    y = _kv_rows(c, x2, y_top - 10, rows)

    # client fields, like the editable spans on the HTML worksheet
    y -= 6
    for label, value in (('Client', client), ('Purchase order', ''),
                         ('Fabric / garment', '')):
        _text(c, x2, y, label, 9, color=MUTED)
        if value:
            _text(c, x2 + 34 * mm, y, value[:40], 9)
        c.setStrokeColor(LINE)
        c.line(x2 + 32 * mm, y - 1, PAGE_W - MARGIN, y - 1)
        y -= 15

    # colour palette
    y = min(y, y_top - img_h) - 24
    _text(c, MARGIN, y, 'THREAD SEQUENCE', 9, bold=True, color=c._accent)
    y -= 6
    c.setStrokeColor(LINE)
    c.line(MARGIN, y, PAGE_W - MARGIN, y)
    y -= 16
    for i, L in enumerate(layers):
        if y < MARGIN + 24:
            _footer(c)
            c.showPage()
            _apply_theme(c, theme, logo_path)
            _header(c, design_name, 'Thread sequence (continued)')
            y = PAGE_H - MARGIN - 48
        c.setFillColor(HexColor(L['hex']))
        c.setStrokeColor(LINE)
        c.rect(MARGIN, y - 3, 8 * mm, 5 * mm, fill=1)
        _text(c, MARGIN + 11 * mm, y, '#%d   %s   %s' % (i + 1, L.get('name', ''), L['hex']), 9)
        if thread_matches and i < len(thread_matches):
            t = thread_matches[i]
            _text(c, MARGIN + 90 * mm, y,
                  '%s  %s %s' % (t['palette'], t['thread_name'],
                                 ('#' + t['thread_number']) if t['thread_number'] else ''),
                  9, color=MUTED)
        y -= 12 * mm / 3.2

    # quote block (embTools port)
    if quote_params:
        q, qrows = quote_rows(stitches, quote_params)
        _text(c, MARGIN, y, 'QUOTE', 9, bold=True, color=c._accent)
        y -= 6
        c.setStrokeColor(LINE)
        c.line(MARGIN, y, PAGE_W - MARGIN, y)
        y -= 14
        for k, v in qrows:
            _text(c, MARGIN, y, k, 9, color=MUTED)
            _text(c, MARGIN + 120 * mm, y, v, 9, align='right')
            y -= 13
        _text(c, MARGIN, y, 'Total', 10, bold=True)
        _text(c, MARGIN + 120 * mm, y, '$%.2f' % q['total'], 10, bold=True, align='right')

    _footer(c)
    c.showPage()

    # ------------------------------------------- page 2: operator detailed
    _header(c, design_name, 'Embroidery worksheet — operator detailed view')
    blocks = block_stats(pattern)
    y = PAGE_H - MARGIN - 40

    _text(c, MARGIN, y, '#', 8, bold=True, color=MUTED)
    _text(c, MARGIN + 14 * mm, y, 'COLOUR', 8, bold=True, color=MUTED)
    _text(c, MARGIN + 74 * mm, y, 'STITCHES', 8, bold=True, color=MUTED)
    _text(c, MARGIN + 98 * mm, y, 'TIME', 8, bold=True, color=MUTED)
    _text(c, MARGIN + 118 * mm, y, 'STOPS/TRIMS', 8, bold=True, color=MUTED)
    _text(c, MARGIN + 146 * mm, y, 'NOTES', 8, bold=True, color=MUTED)
    y -= 5
    c.setStrokeColor(LINE)
    c.line(MARGIN, y, PAGE_W - MARGIN, y)
    y -= 16

    for i, b in enumerate(blocks):
        if y < MARGIN + 20:
            _footer(c)
            c.showPage()
            _header(c, design_name, 'Operator detailed view (continued)')
            y = PAGE_H - MARGIN - 40
        L = layers[i] if i < len(layers) else {'hex': '#888888', 'name': 'Colour %d' % (i + 1)}
        col = HexColor(L['hex'])
        c.setFillColor(col)
        c.setStrokeColor(LINE)
        c.rect(MARGIN, y - 4, 10 * mm, 7 * mm, fill=1)
        # readable index on the swatch, like Ink/Stitch's font_color logic
        lum = (col.red * 0.2126 + col.green * 0.7152 + col.blue * 0.0722)
        c.setFillColor(white if lum < 0.5 else black)
        c.setFont('Helvetica-Bold', 8)
        c.drawCentredString(MARGIN + 5 * mm, y - 1, str(i + 1))

        name = L.get('name', '')
        if thread_matches and i < len(thread_matches):
            t = thread_matches[i]
            name += '  ·  %s %s' % (t['thread_name'],
                                    ('#' + t['thread_number']) if t['thread_number'] else '')
        _text(c, MARGIN + 14 * mm, y, name[:44], 9)
        _text(c, MARGIN + 14 * mm, y - 9, L['hex'], 7, color=MUTED)
        _text(c, MARGIN + 74 * mm, y, '{:,}'.format(b['stitches']), 9)
        _text(c, MARGIN + 98 * mm, y, _est_time(b['stitches'], spm), 9)
        _text(c, MARGIN + 118 * mm, y, '%d / %d' % (b['stops'], b['trims']), 9)
        c.setStrokeColor(LINE)
        c.line(MARGIN + 146 * mm, y - 2, PAGE_W - MARGIN, y - 2)
        y -= 13 * mm / 1.6

    _footer(c)
    c.showPage()
    c.save()
    return path
