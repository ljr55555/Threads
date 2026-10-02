"""StitchForge — upload an image (or type text), get machine-ready embroidery.

Digitizing pipeline from the original StitchForge core, with the Ink/Stitch
family of projects integrated:

* pystitch (vendored)      all format I/O: DST plus PES/EXP/JEF/VP3/XXX/...
* inkstitch                stitch plan SVG + realistic preview, print
                           worksheet PDF, thread palettes
* embroidery-fonts         lettering with pre-digitized satin fonts
* embTools                 quote sheet, stitch player, unit conversion
* svg.panzoom.js           pan/zoom for the previews (served from static/)
"""
import base64
import json
import os
import sys
import tempfile
import traceback
import uuid
from typing import List

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(APP_DIR, 'third_party'))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
import pystitch

from digitizer import segment, engine, render, core
from inkstitchlib import stitch_svg, threads, lettering, worksheet, svginput, business, pen
from inkstitchlib import layers as veclayers
from inkstitchlib import ai_naming
from inkstitchlib import density as density_map
from inkstitchlib import presets, sketch as sketch_mod

#JOBS = os.path.join(tempfile.gettempdir(), 'stitchforge_jobs')
#os.makedirs(JOBS, exist_ok=True)
from pathlib import Path

JOBS = Path(__file__).resolve().parent / "stitchforge_jobs"
JOBS.mkdir(parents=True, exist_ok=True)


app = FastAPI(title='StitchForge')

# big SVG/JSON payloads (stitch plans, realistic SVG, stitch lists) compress
# 4-6x; makes remote deploys and CDN caching worthwhile
app.add_middleware(GZipMiddleware, minimum_size=1024)

# job artifacts never change once written (a new digitize gets a new job id),
# so browsers and any CDN in front (e.g. Cloudflare) may cache them hard
CACHED_PREFIXES = ('/static/', '/api/plan/', '/api/density/', '/api/stitches/',
                   '/api/fonts/', '/api/fill_preview/')


@app.middleware('http')
async def cache_headers(request: Request, call_next):
    response = await call_next(request)
    if (request.method == 'GET' and response.status_code == 200
            and request.url.path.startswith(CACHED_PREFIXES)):
        response.headers.setdefault('Cache-Control', 'public, max-age=86400')
    return response


PATTERNS = {}          # job id -> live pystitch.EmbPattern

EXPORT_FORMATS = {
    'dst': pystitch.write_dst, 'pes': pystitch.write_pes,
    'exp': pystitch.write_exp, 'jef': pystitch.write_jef,
    'vp3': pystitch.write_vp3, 'xxx': pystitch.write_xxx,
    'u01': pystitch.write_u01, 'pec': pystitch.write_pec,
    'tbf': pystitch.write_tbf, 'csv': pystitch.write_csv,
    'json': pystitch.write_json, 'txt': pystitch.write_txt,
    'gcode': pystitch.write_gcode, 'png': pystitch.write_png,
}


def _native(o):
    """NumPy scalars aren't JSON-serialisable; convert on the way out."""
    if isinstance(o, dict):
        return {k: _native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_native(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def _b64(path):
    with open(path, 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()


def _job_dir(job, must_exist=True):
    d = os.path.join(JOBS, job)
    if must_exist and not os.path.isdir(d):
        raise HTTPException(404, 'unknown job')
    return d


def _load_pattern(job):
    """Live pattern if we have it, else re-read the DST (threads from meta)."""
    if job in PATTERNS:
        return PATTERNS[job]
    d = _job_dir(job)
    dst = os.path.join(d, 'design.dst')
    if not os.path.exists(dst):
        raise HTTPException(404, 'no digitized design for this job')
    pat = pystitch.read(dst)
    meta = _load_meta(job)
    pat.threadlist.clear()
    for L in meta.get('layers', []):
        t = pystitch.EmbThread()
        t.color = int(L['hex'].lstrip('#'), 16)
        t.description = L.get('name', '')
        pat.add_thread(t)
    PATTERNS[job] = pat
    return pat


def _load_meta(job):
    p = os.path.join(_job_dir(job), 'meta.json')
    if not os.path.exists(p):
        raise HTTPException(404, 'job has not been digitized yet')
    with open(p) as f:
        return json.load(f)


def _store(job, pat, layers, report, settings, kind='image'):
    d = _job_dir(job, must_exist=False)
    os.makedirs(d, exist_ok=True)
    PATTERNS[job] = pat
    pystitch.write_dst(pat, os.path.join(d, 'design.dst'))
    png = render.preview(pat, [tuple(L['rgb']) for L in layers], os.path.join(d, 'preview.png'))
    with open(os.path.join(d, 'plan.svg'), 'w') as f:
        f.write(stitch_svg.render(pat, realistic=False))
    # a design extended in place (text / SVG added) must not serve old renders
    for stale in ('realistic.svg', 'density.png', 'design.zip'):
        try:
            os.remove(os.path.join(d, stale))
        except FileNotFoundError:
            pass
    meta = {'kind': kind, 'report': report, 'settings': settings,
            'layers': [{'name': L['name'], 'hex': L['hex'], 'rgb': list(L['rgb'])}
                       for L in layers]}
    with open(os.path.join(d, 'meta.json'), 'w') as f:
        json.dump(_native(meta), f)
    return png


# ------------------------------------------- combining designs + page frame
# A design's "frame" records where its page (the SVG artboard it came from)
# sits in pattern coordinates, so SVGs exported from the same artboard can be
# added later and land exactly where they were drawn:
#   settings['frame'] = {'origin': [x, y] (0.1 mm, page top-left),
#                        'scale': size multiplier, 'page': [w_mm, h_mm]}
SEW_CMDS = (pystitch.STITCH, pystitch.JUMP, pystitch.TRIM)


def _extent(pat):
    pts = [(x, y) for x, y, c in pat.stitches
           if (c & 0xFF) in (pystitch.STITCH, pystitch.JUMP)]
    if not pts:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _recenter(pat, frame=None):
    """Move the design's centre to the origin, carrying the frame along."""
    ext = _extent(pat)
    if ext is None:
        return frame
    cx = round((ext[0] + ext[2]) / 2.0)
    cy = round((ext[1] + ext[3]) / 2.0)
    pat.translate(-cx, -cy)
    if frame:
        frame = dict(frame)
        frame['origin'] = [frame['origin'][0] - cx, frame['origin'][1] - cy]
    return frame


def _place_offset(base, add, placement, gap_mm):
    """Offset that puts `add` below/above/left/right of/centred on `base`."""
    b, a = _extent(base), _extent(add)
    if b is None:
        raise HTTPException(400, 'the current design has no stitches to add to')
    if a is None:
        raise HTTPException(400, 'nothing to add')
    gap = gap_mm * 10.0
    aw, ah = a[2] - a[0], a[3] - a[1]
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    if placement == 'below':
        cy = b[3] + gap + ah / 2
    elif placement == 'above':
        cy = b[1] - gap - ah / 2
    elif placement == 'left':
        cx = b[0] - gap - aw / 2
    elif placement == 'right':
        cx = b[2] + gap + aw / 2
    elif placement != 'center':
        raise HTTPException(400, 'placement must be new, keep, below, above, '
                                 'left, right or center')
    return cx - acx, cy - acy


def _merge(base, add, dx=0.0, dy=0.0, first_block=False):
    """Append every colour block of `add` to `base`, shifted by (dx, dy).
    With first_block the base is empty and add's first thread starts it."""
    base.stitches = [q for q in base.stitches if (q[2] & 0xFF) != pystitch.END]
    for i, th in enumerate(add.threadlist):
        base.add_thread(th)
    if not first_block:
        base.color_change()
    for x, y, c in add.stitches:
        k = c & 0xFF
        if k in SEW_CMDS or k == pystitch.COLOR_CHANGE:
            base.add_stitch_absolute(k, x + dx, y + dy)
    base.end()
    return base


def _job_layers(meta):
    layers = meta['layers']
    for L in layers:
        L['rgb'] = tuple(L['rgb'])
    return layers


def basic_report(pat):
    """Size/count report for patterns without source masks (lettering)."""
    pts = [(x, y) for x, y, c in pat.stitches if (c & 0xFF) in (pystitch.STITCH, pystitch.JUMP)]
    a = np.array(pts) / 10.0 if pts else np.zeros((1, 2))
    w = float(a[:, 0].max() - a[:, 0].min())
    h = float(a[:, 1].max() - a[:, 1].min())
    lens, prev = [], None
    trims = jumps = 0
    for x, y, c in pat.stitches:
        k = c & 0xFF
        if k == pystitch.STITCH:
            if prev is not None:
                lens.append(float(np.hypot(x - prev[0], y - prev[1]) / 10))
            prev = (x, y)
        elif k == pystitch.JUMP:
            jumps += 1
            prev = (x, y)
        elif k == pystitch.TRIM:
            trims += 1
            prev = None
        else:
            prev = None
    n_st = sum(1 for q in pat.stitches if (q[2] & 0xFF) == pystitch.STITCH)
    return {'width_mm': round(w, 2), 'height_mm': round(h, 2),
            'width_in': round(w / 25.4, 3), 'stitches': n_st,
            'colour_changes': pat.count_color_changes(),
            'travels': jumps, 'trimmed': trims, 'floats': max(0, jumps - trims),
            'min_stitch_mm': round(min(lens), 2) if lens else 0.0,
            'max_stitch_mm': round(max(lens), 2) if lens else 0.0,
            'runtime_min': round(n_st / 700.0, 1)}


@app.get('/', response_class=HTMLResponse)
def index():
    """Main page: clients and their designs."""
    with open(os.path.join(APP_DIR, 'static', 'clients.html')) as f:
        return f.read()


@app.get('/studio', response_class=HTMLResponse)
def studio():
    """The digitizing studio (canvas, Create/Text/Pen/Design/Export)."""
    with open(os.path.join(APP_DIR, 'static', 'index.html')) as f:
        return f.read()


# ------------------------------------------------------------------- image
@app.post('/api/analyze')
async def analyze(image: UploadFile = File(...), colors: int = Form(3)):
    """Separate the image into colour layers and report stroke widths."""
    if not 1 <= colors <= segment.MAX_COLORS:
        raise HTTPException(400, 'colors must be between 1 and %d' % segment.MAX_COLORS)
    job = uuid.uuid4().hex[:12]
    d = _job_dir(job, must_exist=False)
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, 'src' + os.path.splitext(image.filename or '.png')[1])
    with open(src, 'wb') as f:
        f.write(await image.read())
    try:
        rgba = segment.load(src)
        layers = segment.quantize(rgba, colors)
    except Exception as e:
        raise HTTPException(400, str(e))

    out = []
    for L in layers:
        scale = 100.0 / rgba.shape[1]
        mx, med = segment.stroke_stats(L['mask'], scale)
        out.append({'name': L['name'], 'hex': L['hex'], 'order': L['order'],
                    'area_px': L['area_px'],
                    'max_stroke_at_100mm': round(mx, 2),
                    'typ_stroke_at_100mm': round(med, 2)})
    return _native({'job': job, 'layers': out})


@app.post('/api/digitize')
async def digitize(job: str = Form(...), colors: int = Form(3),
                   width_mm: float = Form(100.0),
                   hoop_w: float = Form(100.0), hoop_h: float = Form(100.0),
                   density: float = Form(0.35), max_satin: float = Form(8.0),
                   heavy_underlay: bool = Form(True),
                   autotune: bool = Form(True),
                   fill_method: str = Form('tatami'),
                   fill_angle: float = Form(65.0),
                   palette: str = Form('Madeira Rayon'),
                   underlay: str = Form(''),
                   pull_comp: float = Form(-1.0),
                   knockdown: bool = Form(False),
                   cap_mode: bool = Form(False),
                   trim_dist: float = Form(2.0),
                   fabric: str = Form(''), machine: str = Form('')):
    if not 1 <= colors <= segment.MAX_COLORS:
        raise HTTPException(400, 'colors must be between 1 and %d' % segment.MAX_COLORS)
    d = _job_dir(job)
    src = None
    for fn in os.listdir(d):
        if fn.startswith('src'):
            src = os.path.join(d, fn)
    if not src:
        raise HTTPException(404, 'upload not found — please re-upload the image')

    try:
        rgba = segment.load(src)
        layers = segment.quantize(rgba, colors)
        if fill_method not in ('tatami', 'contour', 'circular'):
            raise HTTPException(400, 'fill_method must be tatami, contour or circular')
        p = engine.Params(target_width_mm=width_mm, hoop_w_mm=hoop_w,
                          hoop_h_mm=hoop_h, row_spacing=density,
                          satin_spacing=round(density / 2, 4),
                          max_satin=max_satin, heavy_underlay=heavy_underlay,
                          fill_method=fill_method, fill_angle=fill_angle,
                          underlay=presets.clean_underlay(
                              underlay, 'auto' if heavy_underlay else 'light'),
                          pull_comp=presets.clamp(pull_comp, 0.0, 1.0, 0.25)
                          if pull_comp >= 0 else 0.25,
                          knockdown=knockdown, cap_mode=cap_mode,
                          trim_dist=presets.clamp(trim_dist, 0.5, 20.0, 2.0))
        pat, rep, stats, log = engine.digitize(
            layers, p, max_passes=4 if autotune else 2)
        if knockdown:
            # the knockdown block sews first, in the first thread
            layers = [{'name': 'Knockdown', 'hex': layers[0]['hex'],
                       'rgb': layers[0]['rgb']}] + list(layers)
            rep['stitches'] = sum(1 for q in pat.stitches
                                  if (q[2] & 0xFF) == pystitch.STITCH)
        settings = {'density': p.row_spacing, 'max_satin': p.max_satin,
                    'min_fill_area': p.min_fill_area, 'underlay': p.underlay,
                    'pull_comp': p.pull_comp, 'knockdown': knockdown,
                    'cap_mode': cap_mode, 'fabric': fabric, 'machine': machine}
        png = _store(job, pat, layers, rep, settings)
        thread_matches = threads.match_layers(layers, palette)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f'digitizing failed: {e}')

    warnings = []
    narrow = sum(st.get('narrow', 0) for st in stats)
    if narrow:
        warnings.append('%d detail%s narrower than %.1f mm sew as a running stitch '
                        'instead of satin — enlarge the design or simplify the art '
                        'to keep them as satin.' % (narrow, '' if narrow == 1 else 's',
                                                    p.min_satin))
    if not rep['fits_hoop']:
        warnings.append('Design is %.1f x %.1f mm and will not fit the %.0f x %.0f mm '
                        'hoop. Reduce the width.' % (rep['width_mm'], rep['height_mm'],
                                                     hoop_w, hoop_h))
    elif rep['hoop_clearance_mm'] < 2:
        warnings.append('Only %.2f mm of clearance to the hoop edge. Trace the outline '
                        'before stitching.' % rep['hoop_clearance_mm'])
    if rep['max_stitch_mm'] > 10:
        warnings.append('Longest stitch is %.1f mm — long satin snags. Lower the satin '
                        'cap.' % rep['max_stitch_mm'])
    if rep['floats'] > 0:
        warnings.append('%d travel(s) left untrimmed; they will show as short floats.'
                        % rep['floats'])
    if rep['coverage_min'] < 95:
        warnings.append('Lowest layer coverage is %.1f%% — fabric may show through.'
                        % rep['coverage_min'])
    for L in rep['layers']:
        if L['worst_gap_mm2'] > 1.5:
            warnings.append('%s has a %.2f mm2 gap — check that layer in the preview.'
                            % (L['name'], L['worst_gap_mm2']))

    return _native({'job': job, 'kind': 'image', 'report': rep, 'layer_stats': stats,
                    'threads': thread_matches,
                    'passes': [{'pass': e['pass'], 'actions': e['actions'],
                                'width_mm': e['report']['width_mm'],
                                'coverage_min': e['report']['coverage_min'],
                                'stitches': e['report']['stitches']} for e in log],
                    'warnings': warnings, 'preview': _b64(png),
                    'settings': {'density': p.row_spacing, 'max_satin': p.max_satin,
                                 'min_fill_area': p.min_fill_area}})


# --------------------------------------------------------------- lettering
@app.get('/api/fonts')
def fonts():
    return lettering.available_fonts()


@app.get('/api/fonts/{font_id}/preview.png')
def font_preview(font_id: str):
    if '/' in font_id or '..' in font_id:
        raise HTTPException(404, 'no such font')
    p = os.path.join(lettering.FONT_DIR, font_id, 'preview.png')
    if not os.path.exists(p):
        raise HTTPException(404, 'no preview for this font')
    return FileResponse(p, media_type='image/png')


@app.post('/api/lettering')
async def letter(text: str = Form(...), font: str = Form(...),
                 height_mm: float = Form(0.0),
                 letter_spacing_mm: float = Form(0.0),
                 color: str = Form('#1A3B69'),
                 palette: str = Form('Madeira Rayon'),
                 job: str = Form(''),
                 placement: str = Form('new'),
                 gap_mm: float = Form(5.0)):
    text = text.strip('\n')
    if not text.strip():
        raise HTTPException(400, 'please type some text to stitch')
    if font not in {f['id'] for f in lettering.available_fonts()}:
        raise HTTPException(404, 'unknown font')
    try:
        s = core.Sewer()
        rgb_int = int(color.lstrip('#'), 16)
        th = pystitch.EmbThread()
        th.color = rgb_int
        th.description = 'Lettering'
        s.pattern.add_thread(th)
        info = lettering.stitch_text(s, font, text,
                                     height_mm=height_mm or None,
                                     letter_spacing_mm=letter_spacing_mm)
        s.tie_off()
        s.pattern.end()
        pat = s.pattern
        pat.move_center_to_origin()
    except lettering.LetteringError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f'lettering failed: {e}')

    rgb = ((rgb_int >> 16) & 255, (rgb_int >> 8) & 255, rgb_int & 255)
    letter_layer = {'name': 'Lettering', 'hex': '#%06X' % rgb_int, 'rgb': rgb}

    if placement != 'new' and job:
        # add the letters onto the existing design as a new colour block
        base = _load_pattern(job)
        meta = _load_meta(job)
        dx, dy = _place_offset(base, pat, placement if placement != 'keep' else 'center', gap_mm)
        _merge(base, pat, dx, dy)
        settings = dict(meta.get('settings') or {})
        frame = _recenter(base, settings.get('frame'))
        if frame:
            settings['frame'] = frame
        pat = base
        layers = _job_layers(meta) + [letter_layer]
        rep = basic_report(pat)
        png = _store(job, pat, layers, rep, settings, kind=meta['kind'])
    else:
        job = uuid.uuid4().hex[:12]
        layers = [letter_layer]
        rep = basic_report(pat)
        png = _store(job, pat, layers, rep, {'font': font, 'text': text}, kind='lettering')

    warnings = []
    if info['missing_chars']:
        warnings.append('Not in this font, replaced by the default glyph: %s'
                        % ' '.join(info['missing_chars']))
    if height_mm and abs(info['height_mm'] - height_mm) > 0.5:
        warnings.append('This font only scales %.1f–%.1f mm; sewing at %.1f mm.'
                        % (lettering.get_font(font).size * lettering.get_font(font).min_scale,
                           lettering.get_font(font).size * lettering.get_font(font).max_scale,
                           info['height_mm']))

    return _native({'job': job, 'kind': 'lettering', 'report': rep,
                    'lettering': info, 'warnings': warnings,
                    'threads': threads.match_layers(layers, palette),
                    'preview': _b64(png)})


# ------------------------------------------------------------------ import
IMPORT_EXTS = {'.dst', '.pes', '.jef', '.exp', '.vp3', '.xxx', '.u01', '.pec',
               '.tbf', '.hus', '.pcs', '.sew', '.shv', '.jpx', '.dsb', '.dsz',
               '.emd', '.new', '.max', '.mit', '.phb', '.phc', '.stx', '.tap',
               '.txt', '.gt', '.fxy', '.zxy', '.zhs', '.10o', '.100', '.bro',
               '.dat', '.exy', '.inb', '.ksm', '.pcd', '.pcm', '.pcq', '.spx',
               '.stc', '.gcode', '.json', '.csv'}

FALLBACK_COLORS = [(26, 59, 105), (168, 32, 26), (240, 180, 40), (29, 122, 76),
                   (90, 60, 140), (20, 20, 24), (200, 120, 160), (70, 140, 180)]


@app.post('/api/import')
async def import_file(design: List[UploadFile] = File(...),
                      width_mm: float = Form(0.0),
                      fill_method: str = Form('tatami'),
                      fill_angle: float = Form(65.0),
                      density: float = Form(0.35),
                      palette: str = Form('Madeira Rayon'),
                      job: str = Form(''),
                      placement: str = Form('new'),
                      gap_mm: float = Form(5.0),
                      underlay: str = Form(''),
                      pull_comp: float = Form(-1.0),
                      knockdown: bool = Form(False)):
    """Ink/Stitch style input: read any embroidery file, or digitize SVGs.

    Several SVGs in one upload are sewn into one design, each where it sits
    on its page (files exported from the same artboard stay registered).
    With `job` and a placement other than 'new' the result is added onto
    that design: 'keep' puts SVGs at their page position (the page lines up
    with the design's own SVG page, or is centred on the design if it has
    none), below/above/left/right/center place it next to the design."""
    if not design:
        raise HTTPException(400, 'no file uploaded')
    if len(design) > 30:
        raise HTTPException(400, 'too many files (max 30)')
    names = [f.filename or 'design' for f in design]
    exts = [os.path.splitext(n)[1].lower() for n in names]
    if len(design) > 1 and any(e != '.svg' for e in exts):
        raise HTTPException(400, 'several files can only be combined when they '
                                 'are all SVGs — import machine files one at a time')
    adding = bool(job) and placement != 'new'
    base_meta = _load_meta(job) if adding else None
    base_frame = (base_meta.get('settings') or {}).get('frame') if adding else None

    new_job = job if adding else uuid.uuid4().hex[:12]
    svg_warnings = []
    info = {}
    frame = None
    if exts[0] == '.svg':
        # one scale for every file: the design's own frame when adding with
        # 'keep', else the first file's (width_mm applies to the first file)
        scale = base_frame['scale'] if (adding and placement == 'keep' and base_frame) else None
        pat = None
        layers = []
        info = {'elements': 0, 'satin_columns': 0, 'fills': 0, 'strokes': 0,
                'files': []}
        svg_warnings = []
        for f, name in zip(design, names):
            data = await f.read()
            try:
                p, lyr, inf = svginput.digitize_svg(
                    data, width_mm=(width_mm or None) if scale is None else None,
                    scale=scale, fill_method=fill_method, fill_angle=fill_angle,
                    row_spacing=presets.clamp(density, 0.2, 1.0, 0.35), center=False,
                    underlay=presets.clean_underlay(underlay) if underlay else None,
                    pull_comp=presets.clamp(pull_comp, 0.0, 1.0, 0.25) if pull_comp >= 0 else None,
                    knockdown=knockdown)
            except svginput.SvgError as e:
                raise HTTPException(400, '%s: %s' % (name, e))
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(500, f'SVG digitizing failed ({name}): {e}')
            if scale is None:
                scale = inf['scale']
            if frame is None:
                frame = {'origin': [0.0, 0.0], 'scale': scale,
                         'page': [inf['page_w_mm'], inf['page_h_mm']]}
            if pat is None:
                pat = p
            else:
                _merge(pat, p)
            layers += lyr
            for k in ('elements', 'satin_columns', 'fills', 'strokes'):
                info[k] += inf[k]
            info['files'].append(name)
            svg_warnings += [('%s: %s' % (name, w)) if len(names) > 1 else w
                             for w in inf.get('warnings', [])]
            info.setdefault('natural_width_mm', inf['natural_width_mm'])
        if len(names) > 1:
            for i, L in enumerate(layers):
                L['name'] = 'Colour %d' % (i + 1)
        kind = 'svg'
    elif exts[0] in IMPORT_EXTS:
        ext = exts[0]
        data = await design[0].read()
        tmp = os.path.join(_job_dir(new_job, must_exist=False), 'import' + ext)
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, 'wb') as f:
            f.write(data)
        try:
            pat = pystitch.read(tmp)
            if pat is None or not pat.stitches:
                raise ValueError('no stitches found in the file')
        except Exception as e:
            raise HTTPException(400, f'could not read {ext} file: {e}')
        n_blocks = pat.count_color_changes() + 1
        layers = []
        for i in range(n_blocks):
            th = pat.threadlist[i] if i < len(pat.threadlist) else None
            if th is not None and th.color is not None:
                v = th.color & 0xFFFFFF
                rgb = ((v >> 16) & 255, (v >> 8) & 255, v & 255)
            else:
                rgb = FALLBACK_COLORS[i % len(FALLBACK_COLORS)]
            layers.append({'name': (th.description if th is not None and th.description
                                    else 'Colour %d' % (i + 1)),
                           'hex': '#%02X%02X%02X' % rgb, 'rgb': rgb})
        pat.threadlist.clear()
        for L in layers:
            th = pystitch.EmbThread()
            th.color = int(L['hex'][1:], 16)
            th.description = L['name']
            pat.add_thread(th)
        kind = 'import'
        info = {'source_format': ext.lstrip('.'), 'filename': names[0]}
    else:
        raise HTTPException(400, 'unsupported file type %r — upload an SVG or a '
                                 'machine embroidery file' % exts[0])

    if adding:
        base = _load_pattern(job)
        if placement == 'keep':
            if frame is None:
                dx = dy = 0.0      # machine file: its own origin is its home
            elif base_frame:
                dx, dy = base_frame['origin']
            else:
                # the design has no page yet: centre this page on it, and
                # adopt it so later SVGs from the same artboard register
                b = _extent(base)
                if b is None:
                    raise HTTPException(400, 'the current design has no stitches to add to')
                pw, ph = frame['page'][0] * 10.0, (frame['page'][1] or 0) * 10.0
                a = _extent(pat)
                if not ph:
                    ph = (a[1] + a[3]) if a else 0.0
                dx = (b[0] + b[2]) / 2 - pw / 2
                dy = (b[1] + b[3]) / 2 - ph / 2
                base_frame = dict(frame, origin=[dx, dy])
        else:
            dx, dy = _place_offset(base, pat, placement, gap_mm)
            if base_frame is None and frame is not None:
                base_frame = dict(frame, origin=[dx, dy])
        _merge(base, pat, dx, dy)
        settings = dict(base_meta.get('settings') or {})
        base_frame = _recenter(base, base_frame)
        if base_frame:
            settings['frame'] = base_frame
        pat = base
        layers = _job_layers(base_meta) + layers
        kind = base_meta['kind']
        settings_out = settings
    else:
        job = new_job
        frame = _recenter(pat, frame)
        settings_out = dict(info)
        if frame:
            settings_out['frame'] = frame

    rep = basic_report(pat)
    png = _store(job, pat, layers, rep, settings_out, kind=kind)
    return _native({'job': job, 'kind': kind, 'report': rep, 'import_info': info,
                    'threads': threads.match_layers(layers, palette),
                    'warnings': svg_warnings, 'preview': _b64(png)})


def _stitch_settings(data):
    """Sanitised stitch settings shared by the layer and pen paths."""
    st = data.get('settings') or {}
    return {'underlay': presets.clean_underlay(st.get('underlay'), 'auto'),
            'pull_comp': presets.clamp(st.get('pull_comp'), 0.0, 1.0, 0.25),
            'min_satin': presets.clamp(st.get('min_satin'), 0.3, 3.0, 1.0),
            'knockdown': bool(st.get('knockdown')),
            'cap_mode': bool(st.get('cap_mode'))}


@app.get('/api/presets')
def presets_list():
    """Fabric and machine presets the UI applies to the stitch controls."""
    return presets.all_presets()


# ------------------------------------------------- sketch (redwork) mode
@app.post('/api/sketch')
async def sketch_image(image: UploadFile = File(...), width_mm: float = Form(90.0),
                       detail: int = Form(3), color: str = Form('#1A3B69'),
                       bean: bool = Form(True), palette: str = Form('Madeira Rayon')):
    """A photo or line drawing as a one-colour outline sketch (bean stitch)."""
    job = uuid.uuid4().hex[:12]
    d = _job_dir(job, must_exist=False)
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, 'src' + os.path.splitext(image.filename or '.png')[1])
    with open(src, 'wb') as f:
        f.write(await image.read())
    try:
        pat, layers, info = sketch_mod.sketch(
            src, width_mm=max(10.0, min(400.0, width_mm)), detail=detail,
            color=color, bean=bean)
    except sketch_mod.SketchError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f'sketching failed: {e}')
    rep = basic_report(pat)
    png = _store(job, pat, layers, rep, dict(info, width_mm=width_mm), kind='sketch')
    return _native({'job': job, 'kind': 'sketch', 'report': rep, 'sketch_info': info,
                    'threads': threads.match_layers(layers, palette),
                    'warnings': [], 'preview': _b64(png)})


# ------------------------------------------- AI layers (vector-first design)
@app.post('/api/vectorize')
async def vectorize(image: UploadFile = File(...), colors: int = Form(4),
                    width_mm: float = Form(90.0)):
    """Look at the artwork and return editable vector layers — connected
    regions as single objects, similar round shapes grouped, one layer per
    large shape. No stitches are made until /api/stitch_layers."""
    job = uuid.uuid4().hex[:12]
    d = _job_dir(job, must_exist=False)
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, 'src' + os.path.splitext(image.filename or '.png')[1])
    with open(src, 'wb') as f:
        f.write(await image.read())
    try:
        lyrs, w, h = veclayers.vectorize(src, max(1, min(8, colors)),
                                         max(10.0, min(400.0, width_mm)))
    except veclayers.LayerError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f'vectorizing failed: {e}')
    return _native({'layers': lyrs, 'width_mm': round(w, 1), 'height_mm': round(h, 1),
                    'image_job': job, 'ai_names_available': ai_naming.available()})


@app.post('/api/name_layers')
async def name_layers(data: dict):
    """Ask the vision model to name the extracted layers semantically."""
    if not ai_naming.available():
        raise HTTPException(503, 'AI naming needs an Anthropic API key on the '
                                 'server — set ANTHROPIC_API_KEY and restart')
    image_job = str(data.get('image_job') or '')
    lyrs = data.get('layers') or []
    if not lyrs:
        raise HTTPException(400, 'no layers to name')
    d = _job_dir(image_job)
    src = None
    for fn in os.listdir(d):
        if fn.startswith('src'):
            src = os.path.join(d, fn)
    if not src:
        raise HTTPException(404, 'the vectorized image is no longer on the server — re-run AI layers')
    try:
        names = ai_naming.name_layers(src, lyrs)
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f'naming failed: {e}')
    return {'names': {str(k): v for k, v in names.items()}}


@app.post('/api/stitch_layers')
async def stitch_layers(data: dict):
    """Sew the arranged layers into a design (this is when stitches exist)."""
    lyrs = data.get('layers') or []
    if not isinstance(lyrs, list) or not lyrs:
        raise HTTPException(400, 'no layers to stitch')
    if len(lyrs) > 80:
        raise HTTPException(400, 'too many layers (max 80)')
    try:
        pat, block_layers = veclayers.stitch(lyrs, settings=_stitch_settings(data))
    except veclayers.LayerError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f'stitching layers failed: {e}')

    job = uuid.uuid4().hex[:12]
    rep = basic_report(pat)
    png = _store(job, pat, block_layers, rep, {'layers': len(lyrs)}, kind='layers')
    return _native({'job': job, 'kind': 'layers', 'report': rep,
                    'origin_mm': pat.extras.get('origin_mm'),
                    'counts': dict(veclayers.STITCH_COUNTS),
                    'layer_info': {'layers': len(lyrs), 'blocks': len(block_layers)},
                    'threads': threads.match_layers(block_layers,
                                                    data.get('palette', 'Madeira Rayon')),
                    'warnings': [], 'preview': _b64(png)})


@app.get('/api/fills')
def fills_list():
    """The fill patterns, in the order the Available Fills grid shows them."""
    from inkstitchlib import patterns
    return [{'id': k, 'name': patterns.FILL_LABELS[k]} for k in veclayers.FILL_METHODS]


#_FILL_PREVIEW_DIR = os.path.join(tempfile.gettempdir(), 'stitchforge_fills')
_FILL_PREVIEW_DIR = Path(__file__).resolve().parent / "stitchforge_fills"
_FILL_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

@app.get('/api/fill_preview/{name}.png')
def fill_preview(name: str):
    """A swatch of one fill pattern sewn over a 16 mm square."""
    if name not in veclayers.FILL_METHODS:
        raise HTTPException(404, 'no fill named %r' % name)
    os.makedirs(_FILL_PREVIEW_DIR, exist_ok=True)
    out = os.path.join(_FILL_PREVIEW_DIR, name + '.png')
    if not os.path.exists(out):
        sq = {'shell': [[0, 0], [14, 0], [14, 14], [0, 14]], 'holes': []}
        layer = {'name': name, 'color': '#3A78B5', 'polys': [sq],
                 'params': {'stitch': 'fill', 'fill_method': name, 'density': 0.4,
                            'underlay': 'none'}}
        pat, _bl = veclayers.stitch([layer], max_satin=18.0)
        render.preview(pat, [(58, 120, 181)], out, px_wide=300)
    return FileResponse(out, media_type='image/png')


# --------------------------------------------------- pen (manual digitizing)
@app.post('/api/pen')
async def pen_digitize(data: dict):
    """Build a design from shapes traced with the pen tool (points in mm)."""
    shapes = data.get('shapes') or []
    if not isinstance(shapes, list) or not shapes:
        raise HTTPException(400, 'no shapes to stitch')
    if len(shapes) > 200:
        raise HTTPException(400, 'too many shapes (max 200)')
    try:
        pat, layers = pen.build(shapes, settings=_stitch_settings(data))
    except pen.PenError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f'pen digitizing failed: {e}')

    job = uuid.uuid4().hex[:12]
    rep = basic_report(pat)
    png = _store(job, pat, layers, rep,
                 {'shapes': len(shapes)}, kind='pen')
    return _native({'job': job, 'kind': 'pen', 'report': rep,
                    'pen_info': {'shapes': len(shapes), 'blocks': len(layers)},
                    'threads': threads.match_layers(layers, data.get('palette', 'Madeira Rayon')),
                    'warnings': [], 'preview': _b64(png)})


# ----------------------------------------------------------------- exports
ZIP_FORMATS = ('dst', 'pes', 'jef', 'exp', 'vp3', 'xxx')


ORIGINS = ('tl', 'tc', 'tr', 'ml', 'c', 'mr', 'bl', 'bc', 'br')


def _with_origin(pat, origin):
    """A copy of the pattern with the chosen point of its bounding box at
    (0, 0): the machine's origin. 'c' (the centre) is how designs are kept."""
    if origin not in ORIGINS or origin == 'c':
        return pat
    minx, miny, maxx, maxy = pat.bounds()
    col, row = {'l': minx, 'c': (minx + maxx) / 2, 'r': maxx}, {'t': miny, 'm': (miny + maxy) / 2, 'b': maxy}
    if origin == 'c':
        return pat
    ox = col[{'tl': 'l', 'tc': 'c', 'tr': 'r', 'ml': 'l', 'mr': 'r', 'bl': 'l', 'bc': 'c', 'br': 'r'}[origin]]
    oy = row[origin[0]]
    q = pat.copy()
    q.translate(-ox, -oy)
    return q


def _safe_name(name):
    import re
    return re.sub(r'[^\w. -]+', '', str(name or '')).strip()[:80] or 'design'


def _quote_params(kw):
    """Quote parameters from query values; None when no pricing was given."""
    p = {k: kw.get(k) or 0 for k in worksheet.QUOTE_FIELDS}
    if not any(p.values()):
        return None
    return p


@app.get('/api/download/{job}')
def download(job: str, fmt: str = 'dst', formats: str = '', origin: str = 'c', name: str = 'design',
             worksheet_pdf: int = 0, quote_pdf: int = 0, palette: str = 'Madeira Rayon',
             client: str = '', wtheme: int = 0, layout: str = 'production', notes: str = '',
             setup: float = 0.0, price_per_1000: float = 0.0, min_digitizing: float = 0.0,
             garment_qty: int = 0, garment_base: float = 0.0, markup_pct: float = 0.0,
             run_per_1000: float = 0.0, colour_fee: float = 0.0, extra_per_piece: float = 0.0,
             discount_pct: float = 0.0, rush_pct: float = 0.0, tax_pct: float = 0.0):
    """One machine file, or a zip of several formats plus the worksheet and
    quote PDFs. `formats` is a comma list (or 'all'); `fmt=zip` is the old
    export-everything."""
    fmt = fmt.lower()
    name = _safe_name(name)
    wanted = [f.strip().lower() for f in formats.split(',') if f.strip()]
    if fmt == 'zip' or 'all' in wanted:
        wanted = list(ZIP_FORMATS) if (fmt == 'zip' and not wanted) or 'all' in wanted else wanted
        if fmt == 'zip' and not worksheet_pdf and not quote_pdf and not formats:
            worksheet_pdf = 1
    elif not wanted:
        wanted = [fmt]
    bad = [f for f in wanted if f not in EXPORT_FORMATS]
    if bad:
        raise HTTPException(400, 'format must be one of: zip, all, ' + ', '.join(sorted(EXPORT_FORMATS)))
    qp = _quote_params(locals())
    if len(wanted) == 1 and not worksheet_pdf and not quote_pdf:
        f = wanted[0]
        d = _job_dir(job)
        out = os.path.join(d, 'export.' + f)
        pat = _with_origin(_load_pattern(job), origin)
        try:
            EXPORT_FORMATS[f](pat, out)
        except Exception as e:
            raise HTTPException(500, 'export to %s failed: %s' % (f, e))
        return FileResponse(out, filename='%s.%s' % (name, f), media_type='application/octet-stream')
    return _zip_export(job, wanted, origin, name, worksheet_pdf, quote_pdf, qp, palette, client,
                       wtheme, layout, notes)


@app.get('/api/plan/{job}.svg')
def plan_svg(job: str, realistic: bool = False):
    d = _job_dir(job)
    name = 'realistic.svg' if realistic else 'plan.svg'
    p = os.path.join(d, name)
    if not os.path.exists(p):
        pat = _load_pattern(job)
        with open(p, 'w') as f:
            f.write(stitch_svg.render(pat, realistic=realistic))
    return FileResponse(p, media_type='image/svg+xml')


def _theme_of(wtheme):
    if wtheme:
        wt = business.get_wtheme(wtheme)
        if wt:
            return wt['config'], wt.get('logo_path')
    return None, None


def _zip_export(job, formats, origin='c', name='design', worksheet_pdf=1, quote_pdf=0,
                quote_params=None, palette='Madeira Rayon', client='', wtheme=0,
                layout='production', notes=''):
    """Batch export: the chosen formats + worksheet / quote PDFs + thread list in one zip."""
    import zipfile
    d = _job_dir(job)
    pat0 = _load_pattern(job)
    pat = _with_origin(pat0, origin)
    meta = _load_meta(job)
    layers = meta['layers']
    for L in layers:
        L.setdefault('rgb', [int(L['hex'][1:3], 16), int(L['hex'][3:5], 16), int(L['hex'][5:7], 16)])
    try:
        matches = threads.match_layers([{**L, 'rgb': tuple(L['rgb'])} for L in layers], palette)
    except Exception:
        matches = None
    theme, logo_path = _theme_of(wtheme)
    out = os.path.join(d, 'export.zip')
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for fmt in formats:
            p = os.path.join(d, 'export.' + fmt)
            try:
                EXPORT_FORMATS[fmt](pat, p)
                z.write(p, '%s.%s' % (name, fmt))
            except Exception:
                pass
        z.writestr('%s-threads.txt' % name,
                   threads.threadlist_txt(name, meta['report'], layers, matches))
        plan = os.path.join(d, 'plan.svg')
        if os.path.exists(plan):
            z.write(plan, '%s-stitch-plan.svg' % name)
        if worksheet_pdf:
            try:
                pdf = os.path.join(d, 'worksheet.pdf')
                if layout == 'classic':
                    preview, saved_at = os.path.join(d, 'preview.png'), None
                else:
                    preview, saved_at = _sheet_preview(d, pat0, layers)
                worksheet.build(pdf, pat0, meta['report'], layers, thread_matches=matches,
                                preview_png=preview, design_name=name, quote_params=quote_params,
                                client=client, theme=theme, logo_path=logo_path, layout=layout,
                                saved_at=saved_at)
                z.write(pdf, '%s-worksheet.pdf' % name)
            except Exception:
                traceback.print_exc()
        if quote_pdf and quote_params:
            try:
                qpdf = os.path.join(d, 'quote.pdf')
                worksheet.build_quote(qpdf, meta['report'].get('stitches', 0), quote_params,
                                      design_name=name, client=client, theme=theme,
                                      logo_path=logo_path,
                                      colour_changes=meta['report'].get('colour_changes', 0),
                                      notes=notes)
                z.write(qpdf, '%s-quote.pdf' % name)
            except Exception:
                traceback.print_exc()
    return FileResponse(out, filename='%s.zip' % name, media_type='application/zip')


@app.get('/api/quote/{job}.pdf')
def quote_pdf(job: str, name: str = 'design', client: str = '', wtheme: int = 0, inline: bool = False,
              notes: str = '',
              setup: float = 0.0, price_per_1000: float = 0.0, min_digitizing: float = 0.0,
              garment_qty: int = 0, garment_base: float = 0.0, markup_pct: float = 0.0,
              run_per_1000: float = 0.0, colour_fee: float = 0.0, extra_per_piece: float = 0.0,
              discount_pct: float = 0.0, rush_pct: float = 0.0, tax_pct: float = 0.0):
    """The quote alone, as a one-page PDF."""
    meta = _load_meta(job)
    qp = _quote_params(locals()) or {}
    theme, logo_path = _theme_of(wtheme)
    out = os.path.join(_job_dir(job), 'quote.pdf')
    worksheet.build_quote(out, meta['report'].get('stitches', 0), qp, design_name=_safe_name(name),
                          client=client, theme=theme, logo_path=logo_path,
                          colour_changes=meta['report'].get('colour_changes', 0), notes=notes)
    fname = '%s-quote.pdf' % _safe_name(name)
    if inline:
        return FileResponse(out, media_type='application/pdf',
                            headers={'Content-Disposition': 'inline; filename="%s"' % fname})
    return FileResponse(out, filename=fname, media_type='application/pdf')


@app.post('/api/quote_calc')
async def quote_calc(data: dict):
    """Price a design: {stitches, colour_changes, params} -> the breakdown."""
    p = {k: float(v) for k, v in (data.get('params') or {}).items() if k in worksheet.QUOTE_FIELDS}
    return worksheet.quote(data.get('stitches', 0), colour_changes=data.get('colour_changes', 0), **p)


@app.get('/api/quotes')
def quotes_list():
    """Saved quote templates [{id, name, params}]."""
    return business.list_quote_tpls()


@app.post('/api/quotes')
async def quotes_save(data: dict):
    try:
        qid = business.save_quote_tpl(data.get('name'), data.get('params') or {})
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {'id': qid}


@app.delete('/api/quotes/{qid}')
def quotes_delete(qid: int):
    if not business.delete_quote_tpl(qid):
        raise HTTPException(404, 'no such quote template')
    return {'ok': True}


@app.get('/api/stitches/{job}')
def stitches(job: str):
    """Stitch blocks for the front-end stitch player (embTools port)."""
    pat = _load_pattern(job)
    return JSONResponse(stitch_svg.stitch_json(pat))


@app.get('/api/density/{job}.png')
def density_png(job: str):
    """Stitch density heat map (Ink/Stitch density map port)."""
    d = _job_dir(job)
    out = os.path.join(d, 'density.png')
    if not os.path.exists(out):
        pat = _load_pattern(job)
        density_map.render(pat, out)
    return FileResponse(out, media_type='image/png')


@app.get('/api/threadlist/{job}.txt')
def threadlist(job: str, name: str = 'design', palette: str = 'Madeira Rayon'):
    meta = _load_meta(job)
    layers = meta['layers']
    try:
        matches = threads.match_layers(
            [{**L, 'rgb': tuple(L['rgb'])} for L in layers], palette)
    except Exception:
        matches = None
    return Response(threads.threadlist_txt(name, meta['report'], layers, matches),
                    media_type='text/plain')


@app.get('/api/palettes')
def palettes():
    """[{'name', 'custom'}] — built-in Ink/Stitch brands plus user palettes."""
    return threads.available()


@app.get('/api/threads/all')
def threads_all():
    """The whole thread library (every brand) for the Colors panel."""
    return threads.catalog()


@app.get('/api/palettes/{name}')
def palette_detail(name: str):
    try:
        rows = threads.load(name)
    except FileNotFoundError:
        raise HTTPException(404, 'no palette named %r' % name)
    return [{'name': n, 'number': num, 'hex': '#%02X%02X%02X' % (r, g, b)}
            for r, g, b, n, num in rows]


@app.post('/api/palettes')
async def palette_create(data: dict):
    """Add a thread brand: {'name': ..., 'threads': [{'name','number','hex'}]}."""
    try:
        name = threads.save_custom(data.get('name', ''), data.get('threads', []))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {'name': name, 'colors': len(threads.load(name))}


@app.delete('/api/palettes/{name}')
def palette_delete(name: str):
    if not threads.delete_custom(name):
        raise HTTPException(404, 'no custom palette named %r (built-in palettes '
                                 'cannot be deleted)' % name)
    return {'ok': True}


@app.get('/api/match/{job}')
def rematch(job: str, palette: str = 'Madeira Rayon'):
    """Re-match a job's colour layers against another thread palette."""
    meta = _load_meta(job)
    layers = [{**L, 'rgb': tuple(L['rgb'])} for L in meta['layers']]
    return threads.match_layers(layers, palette)


@app.post('/api/recolor/{job}')
async def recolor(job: str, data: dict):
    """Change the design's thread colours (from the Design panel) and persist
    them everywhere: exports, preview, stitch plan, worksheet."""
    import re as _re
    d = _job_dir(job)
    meta = _load_meta(job)
    layers = meta['layers']
    new = data.get('layers') or []
    if len(new) != len(layers):
        raise HTTPException(400, 'expected %d colours (one per colour block), got %d'
                            % (len(layers), len(new)))
    pat = _load_pattern(job)
    for L, n in zip(layers, new):
        hexv = str(n.get('hex', '')).lstrip('#')
        if not _re.fullmatch(r'[0-9a-fA-F]{6}', hexv):
            raise HTTPException(400, 'colours must be 6-digit hex values')
        L['hex'] = '#' + hexv.upper()
        v = int(hexv, 16)
        L['rgb'] = [(v >> 16) & 255, (v >> 8) & 255, v & 255]
        if n.get('name'):
            L['name'] = str(n['name'])[:64]
    pat.threadlist.clear()
    for L in layers:
        th = pystitch.EmbThread()
        th.color = int(L['hex'][1:], 16)
        th.description = L.get('name', '')
        pat.add_thread(th)

    png = render.preview(pat, [tuple(L['rgb']) for L in layers],
                         os.path.join(d, 'preview.png'))
    with open(os.path.join(d, 'plan.svg'), 'w') as f:
        f.write(stitch_svg.render(pat, realistic=False))
    for stale in ('realistic.svg', 'design.zip'):
        p = os.path.join(d, stale)
        if os.path.exists(p):
            os.remove(p)
    with open(os.path.join(d, 'meta.json'), 'w') as f:
        json.dump(_native(meta), f)

    palette = data.get('palette', 'Madeira Rayon')
    matches = threads.match_layers([{**L, 'rgb': tuple(L['rgb'])} for L in layers], palette)
    return _native({'layers': layers, 'preview': _b64(png), 'threads': matches})


# ------------------------------------- worksheet appearance themes (Design panel)
@app.get('/api/wthemes')
def wthemes_list():
    return business.list_wthemes()


@app.post('/api/wthemes')
async def wtheme_save(name: str = Form(...), config: str = Form('{}'),
                      wid: int = Form(0), logo: UploadFile = File(None)):
    try:
        cfg = json.loads(config)
    except Exception:
        raise HTTPException(400, 'config must be JSON')
    logo_bytes = await logo.read() if logo is not None else None
    if logo_bytes and len(logo_bytes) > 4 * 1024 * 1024:
        raise HTTPException(400, 'logo must be under 4 MB')
    new_id = business.save_wtheme(name, cfg, wid or None, logo_bytes)
    if new_id is None:
        raise HTTPException(404, 'no such worksheet theme')
    return {'id': new_id}


@app.delete('/api/wthemes/{wid}')
def wtheme_delete(wid: int):
    if not business.delete_wtheme(wid):
        raise HTTPException(404, 'no such worksheet theme')
    return {'ok': True}


@app.get('/api/wthemes/{wid}/logo')
def wtheme_logo(wid: int):
    t = business.get_wtheme(wid)
    if not t or not t.get('logo_path'):
        raise HTTPException(404, 'this theme has no logo')
    return FileResponse(t['logo_path'], media_type='image/png')


# ------------------------------------------------- design themes (colourways)
@app.get('/api/themes')
def themes_list():
    return business.list_themes()


@app.post('/api/themes')
async def themes_add(data: dict):
    try:
        tid = business.add_theme(data.get('name', ''), data.get('colors') or [])
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {'id': tid}


@app.delete('/api/themes/{tid}')
def themes_delete(tid: int):
    if not business.delete_theme(tid):
        raise HTTPException(404, 'no such theme')
    return {'ok': True}


# ------------------------------------------------ clients & design library
@app.get('/api/clients')
def clients_summary():
    """Clients with design counts (the main page's list)."""
    return business.client_summaries()


@app.get('/api/designs')
def designs_list(client_id: int = 0, unassigned: bool = False):
    return business.list_designs(client_id or None, unassigned=unassigned)


@app.post('/api/designs')
async def designs_save(data: dict):
    """Save the current job into the library, for a client (or unassigned).
    With `design_id` the saved design is overwritten (Save), else a new one
    is made (Save as)."""
    job = str(data.get('job') or '')
    d = _job_dir(job)
    meta = _load_meta(job)
    pat = _load_pattern(job)
    rep = meta.get('report') or basic_report(pat)
    summary = {'kind': meta.get('kind', ''), 'stitches': rep.get('stitches', 0),
               'width_mm': rep.get('width_mm', 0), 'height_mm': rep.get('height_mm', 0),
               'colors': [L['hex'] for L in meta.get('layers', [])]}
    try:
        did = business.save_design(d, data.get('name'), data.get('client_id'),
                                   int(data.get('design_id') or 0) or None, summary)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except KeyError:
        raise HTTPException(404, 'unknown design')
    return business.get_design(did)


def _design_or_404(did):
    dsg = business.get_design(did)
    if not dsg:
        raise HTTPException(404, 'unknown design')
    return dsg


@app.put('/api/designs/{did}')
async def designs_update(did: int, data: dict):
    _design_or_404(did)
    try:
        business.update_design(did, data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return business.get_design(did)


@app.delete('/api/designs/{did}')
def designs_delete(did: int):
    if not business.delete_design(did):
        raise HTTPException(404, 'unknown design')
    return {'ok': True}


@app.get('/api/designs/{did}/preview.png')
def designs_preview(did: int):
    _design_or_404(did)
    p = os.path.join(business.design_dir(did), 'preview.png')
    if not os.path.exists(p):
        raise HTTPException(404, 'no preview')
    return FileResponse(p, media_type='image/png',
                        headers={'Cache-Control': 'no-cache'})


@app.post('/api/designs/{did}/open')
def designs_open(did: int, palette: str = 'Madeira Rayon'):
    """Restore a saved design into a working job (the studio's payload)."""
    import shutil
    dsg = _design_or_404(did)
    src = business.design_dir(did)
    if not os.path.exists(os.path.join(src, 'design.dst')):
        raise HTTPException(404, 'the design files are missing from the library')
    job = uuid.uuid4().hex[:12]
    d = _job_dir(job, must_exist=False)
    os.makedirs(d, exist_ok=True)
    for fn in business.DESIGN_FILES:
        if os.path.exists(os.path.join(src, fn)):
            shutil.copyfile(os.path.join(src, fn), os.path.join(d, fn))
    meta = _load_meta(job)
    pat = _load_pattern(job)
    with open(os.path.join(d, 'plan.svg'), 'w') as f:
        f.write(stitch_svg.render(pat, realistic=False))
    layers = _job_layers(meta)
    # re-render the preview (older saves baked a grey background into it)
    # and refresh the library copy so the clients page picks it up too
    render.preview(pat, [tuple(L['rgb']) for L in layers], os.path.join(d, 'preview.png'))
    shutil.copyfile(os.path.join(d, 'preview.png'), os.path.join(src, 'preview.png'))
    client = business.get_contact('client', dsg['client_id']) if dsg['client_id'] else None
    return _native({'job': job, 'kind': meta.get('kind', 'import'),
                    'report': meta.get('report') or basic_report(pat),
                    'settings': meta.get('settings') or {},
                    'threads': threads.match_layers(layers, palette),
                    'warnings': [], 'preview': _b64(os.path.join(d, 'preview.png')),
                    'design': dsg, 'client': client})


# ------------------------------------------------------------------ hoops
@app.get('/api/hoops')
def hoops_list():
    return business.list_hoops()


@app.post('/api/hoops')
async def hoops_add(data: dict):
    try:
        hid = business.add_hoop(data.get('name'), data.get('w_mm'), data.get('h_mm'))
    except (TypeError, ValueError) as e:
        raise HTTPException(400, str(e) if str(e) else 'w_mm and h_mm are required')
    return next(h for h in business.list_hoops() if h['id'] == hid)


@app.delete('/api/hoops/{hid}')
def hoops_delete(hid: int):
    if not business.delete_hoop(hid):
        raise HTTPException(404, 'unknown hoop')
    return {'ok': True}


# --------------------------------------------- business database (embTools)
@app.get('/api/business/{kind}')
def business_list(kind: str, sort: str = 'name'):
    if kind not in business.KINDS:
        raise HTTPException(404, 'kind must be client or vendor')
    return business.list_contacts(kind, sort)


@app.post('/api/business/{kind}')
async def business_add(kind: str, data: dict):
    if kind not in business.KINDS:
        raise HTTPException(404, 'kind must be client or vendor')
    if not str(data.get('name', '')).strip():
        raise HTTPException(400, 'name is required')
    return {'id': business.add_contact(kind, data)}


@app.put('/api/business/{kind}/{cid}')
async def business_update(kind: str, cid: int, data: dict):
    if kind not in business.KINDS:
        raise HTTPException(404, 'kind must be client or vendor')
    if not business.update_contact(kind, cid, data):
        raise HTTPException(404, 'no such contact')
    return {'ok': True}


@app.delete('/api/business/{kind}/{cid}')
def business_delete(kind: str, cid: int):
    if kind not in business.KINDS:
        raise HTTPException(404, 'kind must be client or vendor')
    if not business.delete_contact(kind, cid):
        raise HTTPException(404, 'no such contact')
    return {'ok': True}


@app.get('/api/notes/{kind}')
def notes_get(kind: str):
    if kind not in business.NOTE_KINDS:
        raise HTTPException(404, 'kind must be notes, quotes or todo')
    return {'kind': kind, 'text': business.get_note(kind)}


@app.put('/api/notes/{kind}')
async def notes_set(kind: str, data: dict):
    if kind not in business.NOTE_KINDS:
        raise HTTPException(404, 'kind must be notes, quotes or todo')
    business.set_note(kind, data.get('text', ''))
    return {'ok': True}


def _sheet_preview(d, pat, layers):
    """Production worksheet artwork: printed on white with shaded strands,
    plus the time the design was last saved."""
    import datetime
    out = os.path.join(d, 'worksheet_preview.png')
    render.preview(pat, [tuple(L['rgb']) for L in layers], out,
                   px_wide=1600, bg=(255, 255, 255), shade=True)
    saved = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(d, 'meta.json')))
    return out, saved


@app.get('/api/worksheet/{job}.pdf')
def worksheet_pdf(job: str, name: str = 'design', palette: str = 'Madeira Rayon',
                  client: str = '', inline: bool = False, wtheme: int = 0,
                  setup: float = 0.0, price_per_1000: float = 0.0,
                  garment_qty: int = 0, garment_base: float = 0.0,
                  markup_pct: float = 0.0, discount_pct: float = 0.0,
                  min_digitizing: float = 0.0, run_per_1000: float = 0.0, colour_fee: float = 0.0,
                  extra_per_piece: float = 0.0, rush_pct: float = 0.0, tax_pct: float = 0.0,
                  layout: str = 'production', title: str = ''):
    d = _job_dir(job)
    pat = _load_pattern(job)
    meta = _load_meta(job)
    layers = meta['layers']
    for L in layers:
        L.setdefault('rgb', [int(L['hex'][1:3], 16), int(L['hex'][3:5], 16), int(L['hex'][5:7], 16)])
        L['rgb'] = tuple(L['rgb'])
    try:
        matches = threads.match_layers(layers, palette)
    except Exception:
        matches = None
    quote_params = _quote_params(locals())
    if quote_params:
        quote_params['colour_changes'] = meta['report'].get('colour_changes', 0)
    theme, logo_path = _theme_of(wtheme)
    out = os.path.join(d, 'worksheet.pdf')
    if layout == 'classic':
        preview, saved_at = os.path.join(d, 'preview.png'), None
    else:
        preview, saved_at = _sheet_preview(d, pat, layers)
    worksheet.build(out, pat, meta['report'], layers, thread_matches=matches,
                    preview_png=preview,
                    design_name=name or 'design', quote_params=quote_params,
                    client=client, theme=theme, logo_path=logo_path,
                    layout=layout, saved_at=saved_at, title=title)
    fname = '%s-worksheet.pdf' % (name or 'design')
    if inline:
        # render in the browser's PDF viewer (the UI's preview modal)
        return FileResponse(out, media_type='application/pdf',
                            headers={'Content-Disposition': 'inline; filename="%s"' % fname})
    return FileResponse(out, filename=fname, media_type='application/pdf')


app.mount('/static', StaticFiles(directory=os.path.join(APP_DIR, 'static')),
          name='static')
