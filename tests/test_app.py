"""End-to-end tests over the HTTP API (run with: pytest)."""
import io
import os
import re
import sys
import tempfile

import numpy as np
import pytest
from PIL import Image, ImageDraw

os.environ.setdefault('STITCHFORGE_DATA',
                      os.path.join(tempfile.mkdtemp(), 'sf_test_data'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
import pystitch
from app import app

client = TestClient(app)


def _test_png():
    im = Image.new('RGBA', (400, 260), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([30, 30, 220, 220], fill=(26, 59, 105, 255))
    d.ellipse([80, 80, 170, 170], fill=(0, 0, 0, 0))
    d.rounded_rectangle([250, 60, 380, 120], 18, fill=(168, 32, 26, 255))
    buf = io.BytesIO()
    im.save(buf, 'PNG')
    return buf.getvalue()


@pytest.fixture(scope='module')
def image_job():
    r = client.post('/api/analyze',
                    files={'image': ('t.png', _test_png(), 'image/png')},
                    data={'colors': 2})
    assert r.status_code == 200
    job = r.json()['job']
    r = client.post('/api/digitize', data={
        'job': job, 'colors': 2, 'width_mm': 60, 'hoop_w': 100, 'hoop_h': 100,
        'density': 0.4, 'max_satin': 8, 'heavy_underlay': False, 'autotune': False})
    assert r.status_code == 200, r.text
    return job, r.json()


def test_digitize_report(image_job):
    job, d = image_job
    rep = d['report']
    assert 55 <= rep['width_mm'] <= 65
    assert rep['stitches'] > 200
    assert len(d['threads']) == 2
    assert d['threads'][0]['thread_name']


def test_image_quantizer_supports_32_colors():
    from digitizer import segment
    colors = np.array([[i * 7 % 256, i * 13 % 256, i * 23 % 256, 255]
                       for i in range(32)], dtype=np.int16)
    rgba = np.repeat(colors[:, None, :], 20, axis=1)
    layers = segment.quantize(rgba, 32, np.ones(rgba.shape[:2], dtype=bool))
    assert len(layers) == 32


def test_image_color_limit_is_validated():
    r = client.post('/api/analyze',
                    files={'image': ('t.png', _test_png(), 'image/png')},
                    data={'colors': 33})
    assert r.status_code == 400
    assert r.json()['detail'] == 'colors must be between 1 and 32'


def test_exports(image_job):
    job, _ = image_job
    for fmt in ('dst', 'pes', 'jef', 'exp', 'vp3', 'csv'):
        r = client.get('/api/download/%s?fmt=%s' % (job, fmt))
        assert r.status_code == 200 and len(r.content) > 100, fmt
    assert client.get('/api/download/%s?fmt=nope' % job).status_code == 400


def test_svg_and_stitch_json(image_job):
    job, _ = image_job
    r = client.get('/api/plan/%s.svg' % job)
    assert r.status_code == 200 and b'<svg' in r.content
    r = client.get('/api/plan/%s.svg?realistic=true' % job)
    assert r.status_code == 200 and b'realistic-stitch-filter' in r.content
    r = client.get('/api/stitches/%s' % job)
    assert r.status_code == 200 and len(r.json()) >= 1


def test_worksheet_pdf(image_job):
    job, _ = image_job
    r = client.get('/api/worksheet/%s.pdf?name=Test&setup=10&price_per_1000=1.5' % job)
    assert r.status_code == 200
    assert r.content[:4] == b'%PDF'


def test_fonts_listing():
    r = client.get('/api/fonts')
    assert r.status_code == 200
    ids = {f['id'] for f in r.json()}
    assert 'emilio_20' in ids and 'sacramarif' in ids


def test_lettering():
    r = client.post('/api/lettering', data={
        'text': 'Abc', 'font': 'geneva_simple', 'height_mm': 15, 'color': '#8B1A1A'})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['report']['stitches'] > 100
    assert d['lettering']['satin_columns'] > 0
    r = client.get('/api/download/%s?fmt=dst' % d['job'])
    assert r.status_code == 200


def test_lettering_bad_input():
    assert client.post('/api/lettering', data={
        'text': '   ', 'font': 'geneva_simple'}).status_code == 400
    assert client.post('/api/lettering', data={
        'text': 'hi', 'font': 'no_such_font'}).status_code == 404


def test_palettes():
    r = client.get('/api/palettes')
    names = [p['name'] for p in r.json()]
    assert 'Madeira Rayon' in names
    assert 'BAI Matte' in names          # seeded from the product colour card
    assert len(names) > 70               # the full Ink/Stitch palette set
    r = client.get('/api/palettes/BAI Matte')
    rows = r.json()
    assert r.status_code == 200 and len(rows) == 20
    assert any(t['number'] == '8382' for t in rows)


def test_custom_palette_crud_and_match(image_job):
    job, _ = image_job
    r = client.post('/api/palettes', json={'name': 'Test Brand', 'threads': [
        {'name': 'Navy', 'number': 'T1', 'hex': '#1A3B69'},
        {'name': 'Flame', 'number': 'T2', 'hex': '#A8201A'}]})
    assert r.status_code == 200 and r.json()['colors'] == 2
    assert any(p['name'] == 'Test Brand' and p['custom']
               for p in client.get('/api/palettes').json())
    m = client.get('/api/match/%s?palette=Test%%20Brand' % job).json()
    assert all(t['palette'] == 'Test Brand' for t in m)
    assert {t['thread_number'] for t in m} <= {'T1', 'T2'}
    assert client.delete('/api/palettes/Test Brand').status_code == 200
    assert client.delete('/api/palettes/Madeira Rayon').status_code == 404
    assert client.post('/api/palettes', json={'name': 'x', 'threads': []}).status_code == 400


def test_fill_methods():
    for method in ('contour', 'circular'):
        r = client.post('/api/analyze',
                        files={'image': ('t.png', _test_png(), 'image/png')},
                        data={'colors': 2})
        job = r.json()['job']
        r = client.post('/api/digitize', data={
            'job': job, 'colors': 2, 'width_mm': 50, 'hoop_w': 100, 'hoop_h': 100,
            'density': 0.4, 'max_satin': 8, 'heavy_underlay': False,
            'autotune': False, 'fill_method': method})
        assert r.status_code == 200, (method, r.text)
        assert r.json()['report']['stitches'] > 200


TEST_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" width="60mm" height="40mm"
     viewBox="0 0 60 40">
  <rect x="4" y="4" width="24" height="16" fill="#1a3b69"/>
  <circle cx="45" cy="12" r="8" fill="#a8201a"/>
  <path d="M4,30 C20,38 40,22 56,32" fill="none" stroke="#1d7a4c" stroke-width="0.5"/>
</svg>'''


def test_svg_import():
    r = client.post('/api/import',
                    files={'design': ('art.svg', TEST_SVG.encode(), 'image/svg+xml')},
                    data={'width_mm': 0})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['kind'] == 'svg'
    assert d['import_info']['fills'] >= 2
    assert d['import_info']['strokes'] >= 1
    # content spans x=4..56 of the 60mm document -> ~52mm stitched extent
    assert 48 <= d['report']['width_mm'] <= 56
    assert d['report']['stitches'] > 300
    assert client.get('/api/download/%s?fmt=dst' % d['job']).status_code == 200


def test_embroidery_file_import(image_job):
    job, _ = image_job
    dst = client.get('/api/download/%s?fmt=dst' % job).content
    r = client.post('/api/import',
                    files={'design': ('old.dst', dst, 'application/octet-stream')})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['kind'] == 'import'
    assert d['report']['stitches'] > 200
    assert client.get('/api/download/%s?fmt=pes' % d['job']).status_code == 200


def test_density_and_zip_and_threadlist(image_job):
    job, _ = image_job
    r = client.get('/api/density/%s.png' % job)
    assert r.status_code == 200 and r.content[:8].startswith(b'\x89PNG')
    r = client.get('/api/download/%s?fmt=zip' % job)
    assert r.status_code == 200 and r.content[:2] == b'PK'
    r = client.get('/api/threadlist/%s.txt' % job)
    assert r.status_code == 200 and b'Thread order' in r.content


def test_business_crud():
    r = client.post('/api/business/client', json={'name': 'Acme Embroidery',
                                                  'business_name': 'Acme'})
    assert r.status_code == 200
    cid = r.json()['id']
    rows = client.get('/api/business/client').json()
    assert any(row['id'] == cid for row in rows)
    assert client.put('/api/business/client/%d' % cid,
                      json={'name': 'Acme 2'}).status_code == 200
    assert client.delete('/api/business/client/%d' % cid).status_code == 200
    assert client.post('/api/business/client', json={'name': '  '}).status_code == 400
    assert client.get('/api/business/nope').status_code == 404


def test_notes():
    assert client.put('/api/notes/todo', json={'text': 'hoop the caps'}).status_code == 200
    assert client.get('/api/notes/todo').json()['text'] == 'hoop the caps'
    assert client.get('/api/notes/nope').status_code == 404


def test_wthemes_and_themed_worksheet(image_job):
    job, _ = image_job
    r = client.post('/api/wthemes', data={
        'name': 'Letterhead', 'wid': 0,
        'config': '{"accent": "#A8201A", "font": "Times", "footer": "Studio X"}'},
        files={'logo': ('l.png', _test_png(), 'image/png')})
    assert r.status_code == 200
    wid = r.json()['id']
    themes = client.get('/api/wthemes').json()
    assert any(t['id'] == wid and t['has_logo'] and t['config']['font'] == 'Times'
               for t in themes)
    assert client.get('/api/wthemes/%d/logo' % wid).status_code == 200
    r = client.get('/api/worksheet/%s.pdf?wtheme=%d' % (job, wid))
    assert r.status_code == 200 and r.content[:4] == b'%PDF'
    assert client.delete('/api/wthemes/%d' % wid).status_code == 200


def test_lettering_append():
    r = client.post('/api/analyze',
                    files={'image': ('t.png', _test_png(), 'image/png')},
                    data={'colors': 2})
    job = r.json()['job']
    r = client.post('/api/digitize', data={
        'job': job, 'colors': 2, 'width_mm': 50, 'autotune': False,
        'heavy_underlay': False, 'density': 0.4})
    base = r.json()['report']
    r = client.post('/api/lettering', data={
        'text': 'Hi', 'font': 'geneva_simple', 'height_mm': 12,
        'color': '#188652', 'job': job, 'placement': 'below'})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['job'] == job
    assert d['report']['stitches'] > base['stitches']
    assert d['report']['colour_changes'] == base['colour_changes'] + 1
    assert d['report']['height_mm'] > base['height_mm']


def test_pen_digitize():
    shapes = [
        {'mode': 'pairs', 'color': '#A8201A', 'spacing_mm': 0.4,
         'points': [[0, 0], [0, 6], [10, 0.5], [10, 6.5], [20, 0], [20, 6]]},
        {'mode': 'center', 'color': '#1A3B69', 'width_mm': 3, 'spacing_mm': 0.4,
         'points': [[0, 20], [15, 24], [30, 20]]},
        {'mode': 'run', 'color': '#188652', 'points': [[0, 40], [30, 40]]},
    ]
    r = client.post('/api/pen', json={'shapes': shapes})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['kind'] == 'pen'
    assert d['pen_info']['blocks'] == 3
    assert d['report']['stitches'] > 100
    assert client.get('/api/download/%s?fmt=dst' % d['job']).status_code == 200
    assert client.post('/api/pen', json={'shapes': []}).status_code == 400
    assert client.post('/api/pen', json={'shapes': [
        {'mode': 'pairs', 'points': [[0, 0], [1, 1]]}]}).status_code == 400


def test_vectorize_and_stitch_layers():
    import math
    from PIL import Image as PImage, ImageDraw as PDraw
    im = PImage.new('RGBA', (600, 600), (0, 0, 0, 0))
    d = PDraw.Draw(im)
    d.ellipse([80, 80, 520, 520], outline=(26, 59, 105, 255), width=60)
    for k in range(6):
        a = k * math.pi / 3
        x, y = 300 + 270 * math.cos(a), 300 + 270 * math.sin(a)
        d.ellipse([x - 28, y - 28, x + 28, y + 28], fill=(26, 59, 105, 255))
    d.rectangle([250, 260, 350, 340], fill=(240, 180, 40, 255))
    buf = io.BytesIO()
    im.save(buf, 'PNG')

    r = client.post('/api/vectorize',
                    files={'image': ('ring.png', buf.getvalue(), 'image/png')},
                    data={'colors': 2, 'width_mm': 80})
    assert r.status_code == 200, r.text
    lyrs = r.json()['layers']
    names = [L['name'] for L in lyrs]
    # the connected ring is one object; the round heads are grouped
    assert any('round ×' in n for n in names)
    assert any('shape 1' in n for n in names)
    assert all(len(L['polys']) >= 1 for L in lyrs)

    r = client.post('/api/stitch_layers', json={'layers': lyrs})
    assert r.status_code == 200, r.text
    d2 = r.json()
    assert d2['kind'] == 'layers'
    assert d2['report']['stitches'] > 500
    assert client.get('/api/download/%s?fmt=pes' % d2['job']).status_code == 200
    # hiding every layer is an error, not a crash
    for L in lyrs:
        L['visible'] = False
    assert client.post('/api/stitch_layers', json={'layers': lyrs}).status_code == 400


def _ring_png():
    im = Image.new('RGBA', (400, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([100, 100, 340, 340], outline=(26, 59, 105, 255), width=45)
    d.ellipse([10, 10, 70, 70], fill=(26, 59, 105, 255))
    buf = io.BytesIO()
    im.save(buf, 'PNG')
    return buf.getvalue()


def test_ai_naming_unavailable(monkeypatch):
    from inkstitchlib import ai_naming
    monkeypatch.setattr(ai_naming, 'available', lambda: False)
    r = client.post('/api/vectorize',
                    files={'image': ('r.png', _ring_png(), 'image/png')},
                    data={'colors': 1, 'width_mm': 60})
    assert r.status_code == 200
    d = r.json()
    assert d['ai_names_available'] is False
    assert d['image_job']
    r = client.post('/api/name_layers',
                    json={'image_job': d['image_job'], 'layers': d['layers']})
    assert r.status_code == 503
    assert 'ANTHROPIC_API_KEY' in r.json()['detail']


def test_ai_naming_with_mocked_model(monkeypatch):
    from inkstitchlib import ai_naming

    r = client.post('/api/vectorize',
                    files={'image': ('r.png', _ring_png(), 'image/png')},
                    data={'colors': 1, 'width_mm': 60})
    d = r.json()
    lyrs = d['layers']
    ids = [L['id'] for L in lyrs]

    class FakeBlock:
        type = 'text'
        text = 'Here you go: {"%d": "Ring", "%d": "Head"}' % (ids[0], ids[-1])

    class FakeResponse:
        stop_reason = 'end_turn'
        content = [FakeBlock()]

    class FakeMessages:
        def create(self, **kwargs):
            # the request must carry both images and the layer listing
            blocks = kwargs['messages'][0]['content']
            assert sum(1 for b in blocks if b['type'] == 'image') == 2
            assert 'JSON' in blocks[-1]['text']
            return FakeResponse()

    class FakeClient:
        def __init__(self, *a, **k):
            self.messages = FakeMessages()

    import anthropic
    monkeypatch.setattr(ai_naming, 'available', lambda: True)
    monkeypatch.setattr(anthropic, 'Anthropic', FakeClient)

    r = client.post('/api/name_layers',
                    json={'image_job': d['image_job'], 'layers': lyrs})
    assert r.status_code == 200, r.text
    names = r.json()['names']
    assert names[str(ids[0])] == 'Ring'
    assert names[str(ids[-1])] == 'Head'


SVG_A = '''<svg xmlns="http://www.w3.org/2000/svg" width="100mm" height="100mm"
     viewBox="0 0 100 100"><rect x="10" y="10" width="20" height="20" fill="#1a3b69"/></svg>'''
SVG_B = '''<svg xmlns="http://www.w3.org/2000/svg" width="100mm" height="100mm"
     viewBox="0 0 100 100"><rect x="70" y="70" width="20" height="20" fill="#a8201a"/></svg>'''
SVG_AB = SVG_A.replace('</svg>', '<rect x="70" y="70" width="20" height="20" '
                                  'fill="#a8201a"/></svg>')


def _extent_mm(job):
    from app import _load_pattern, _extent
    return [round(v / 10.0) for v in _extent(_load_pattern(job))]


def test_svgs_keep_their_page_position():
    # two SVGs from the same artboard sew exactly like the combined drawing
    r = client.post('/api/import', files=[('design', ('a.svg', SVG_A.encode())),
                                          ('design', ('b.svg', SVG_B.encode()))])
    assert r.status_code == 200, r.text
    assert len(r.json()['threads']) == 2
    both = _extent_mm(r.json()['job'])
    r = client.post('/api/import', files={'design': ('ab.svg', SVG_AB.encode())})
    assert _extent_mm(r.json()['job']) == both == [-40, -40, 40, 40]

    # adding the second file later with 'keep' lands it in the same place
    r = client.post('/api/import', files={'design': ('a.svg', SVG_A.encode())})
    job = r.json()['job']
    assert _extent_mm(job) == [-10, -10, 10, 10]
    r = client.post('/api/import', files={'design': ('b.svg', SVG_B.encode())},
                    data={'job': job, 'placement': 'keep'})
    assert r.status_code == 200, r.text
    assert r.json()['job'] == job
    assert _extent_mm(job) == both
    assert len(r.json()['threads']) == 2


def test_svg_added_beside_a_design(image_job):
    job, d = image_job
    r = client.post('/api/import', files={'design': ('a.svg', SVG_A.encode())},
                    data={'job': job, 'placement': 'right', 'gap_mm': 5})
    assert r.status_code == 200, r.text
    assert r.json()['report']['width_mm'] > d['report']['width_mm'] + 20
    r = client.post('/api/import', files=[('design', ('a.svg', SVG_A.encode())),
                                          ('design', ('x.dst', b'nope'))])
    assert r.status_code == 400


def test_production_worksheet():
    from inkstitchlib import worksheet
    r = client.post('/api/import', files={'design': ('ab.svg', SVG_AB.encode())})
    job = r.json()['job']
    for q in ('', '&layout=classic', '&setup=10&price_per_1000=1.5'):
        r = client.get('/api/worksheet/%s.pdf?name=logo%s' % (job, q))
        assert r.status_code == 200 and r.content[:4] == b'%PDF', q
    from app import _load_pattern
    ps = worksheet.production_stats(_load_pattern(job))
    assert abs(ps['left_mm'] - 40) < 0.5 and abs(ps['down_mm'] - 40) < 0.5
    assert ps['max_stitch_mm'] > 0 and ps['thread_ft'] > ps['bobbin_ft'] > 0


def test_worksheet_thread_lists_paginate(tmp_path, monkeypatch):
    from inkstitchlib import worksheet
    pattern = pystitch.EmbPattern()
    layers = []
    for index in range(40):
        color = ((index * 47) % 256 << 16) | ((index * 83) % 256 << 8) | ((index * 131) % 256)
        thread = pystitch.EmbThread()
        thread.color = color
        pattern.add_thread(thread)
        if index:
            pattern.color_change()
        pattern.stitch_abs(index * 10, 0)
        pattern.stitch_abs(index * 10 + 5, 5)
        layers.append({'name': 'Colour %d' % (index + 1), 'hex': '#%06X' % color})

    drawn = []
    original_text = worksheet._text

    def capture_text(canvas, x, y, text, *args, **kwargs):
        drawn.append(str(text))
        return original_text(canvas, x, y, text, *args, **kwargs)

    monkeypatch.setattr(worksheet, '_text', capture_text)
    report = {'stitches': 80, 'width_mm': 40, 'height_mm': 1,
              'colour_changes': 39, 'travels': 0, 'trimmed': 0}

    production = tmp_path / 'production.pdf'
    worksheet.build(str(production), pattern, report, layers, layout='production')
    assert '40.' in drawn
    assert len(re.findall(rb'/Type\s*/Page\b', production.read_bytes())) == 2

    drawn.clear()
    classic = tmp_path / 'classic.pdf'
    worksheet.build(str(classic), pattern, report, layers, layout='classic')
    assert any(text.startswith('#40 ') for text in drawn)
    assert len(re.findall(rb'/Type\s*/Page\b', classic.read_bytes())) >= 3


def test_client_design_library():
    # main page + studio routes
    assert b'clients' in client.get('/').content.lower()
    assert b'Save to client' in client.get('/studio').content
    cid = client.post('/api/business/client', json={'name': 'Library Co'}).json()['id']
    r = client.post('/api/import', files={'design': ('ab.svg', SVG_AB.encode())})
    job = r.json()['job']
    r = client.post('/api/designs', json={'job': job, 'name': 'Chest logo', 'client_id': cid})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['client_id'] == cid and d['stitches'] > 100 and len(d['colors']) == 2
    summ = {c['id']: c for c in client.get('/api/clients').json()}
    assert summ[cid]['design_count'] == 1
    assert [x['id'] for x in client.get('/api/designs?client_id=%d' % cid).json()] == [d['id']]
    assert client.get('/api/designs/%d/preview.png' % d['id']).status_code == 200

    # overwrite (Save) keeps the id; status/notes update; bad status rejected
    r = client.post('/api/designs', json={'job': job, 'name': 'Chest logo v2',
                                          'client_id': cid, 'design_id': d['id']})
    assert r.json()['id'] == d['id'] and r.json()['name'] == 'Chest logo v2'
    r = client.put('/api/designs/%d' % d['id'], json={'status': 'approved', 'notes': 'navy tee'})
    assert r.json()['status'] == 'approved' and r.json()['notes'] == 'navy tee'
    assert client.put('/api/designs/%d' % d['id'], json={'status': 'nope'}).status_code == 400
    assert client.post('/api/designs', json={'job': job, 'client_id': 99999}).status_code == 400

    # reopen into a fresh working job that exports like the original
    r = client.post('/api/designs/%d/open' % d['id'])
    assert r.status_code == 200, r.text
    o = r.json()
    assert o['job'] != job and o['client']['name'] == 'Library Co'
    assert abs(o['report']['width_mm'] - 80) < 2
    assert client.get('/api/download/%s?fmt=dst' % o['job']).status_code == 200
    assert client.get('/api/worksheet/%s.pdf' % o['job']).status_code == 200

    # deleting the client leaves the design unassigned; deleting the design removes it
    client.delete('/api/business/client/%d' % cid)
    assert d['id'] in [x['id'] for x in client.get('/api/designs?unassigned=true').json()]
    assert client.delete('/api/designs/%d' % d['id']).status_code == 200
    assert client.get('/api/designs/%d/preview.png' % d['id']).status_code == 404


def _import_svg(svg, **data):
    r = client.post('/api/import', files={'design': ('t.svg', svg.encode())}, data=data)
    return r


def test_svg_reader_handles_design_tool_exports():
    """CSS classes, <use x y> placement, gradients, clip-paths, wide strokes,
    paint-order — the constructs Illustrator/Canva/Figma exports rely on."""
    # Illustrator: colours only in a <style> class; a stroke wider than the
    # satin cap is sewn as a fill band under the fill (paint-order: stroke)
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 60">
      <defs><style>.cls-1{fill:#fdf0e1;stroke:#0047ab;stroke-width:12px;paint-order:stroke}
      .cls-2{fill:#0047ab}</style></defs>
      <path class="cls-1" d="M20,10 h40 v20 h-40 z"/>
      <rect class="cls-2" x="20" y="45" width="40" height="8"/></svg>'''
    d = _import_svg(svg).json()
    assert d['kind'] == 'svg', d
    th = {t['hex'].upper() for t in d['threads']}
    assert th == {'#FDF0E1', '#0047AB'}, th
    # the band around the rect extends it by 6 units each side -> 52 of 100 units
    assert 48 <= d['report']['width_mm'] / (d['import_info']['natural_width_mm'] / 100) <= 56

    # Canva/cairo: glyphs in <defs>, placed with <use x y> (translate-only
    # transforms used to be dropped and every glyph landed at the origin)
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
      width="100mm" height="40mm" viewBox="0 0 100 40">
      <defs><path id="g" d="M0,0 h10 v10 h-10 z"/></defs>
      <g fill="#a8201a"><use xlink:href="#g" x="5" y="5"/><use xlink:href="#g" x="85" y="25"/></g></svg>'''
    d = _import_svg(svg).json()
    assert 88 <= d['report']['width_mm'] <= 92 and 28 <= d['report']['height_mm'] <= 32, d['report']

    # gradient fill sews the average stop colour; a clip-path cuts the shape
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="100mm" height="50mm" viewBox="0 0 100 50">
      <defs><linearGradient id="gr"><stop offset="0" stop-color="#000000"/><stop offset="1" stop-color="#0000ff"/></linearGradient>
      <clipPath id="c"><rect x="0" y="0" width="50" height="50"/></clipPath></defs>
      <rect x="10" y="10" width="80" height="30" fill="url(#gr)" clip-path="url(#c)"/></svg>'''
    d = _import_svg(svg).json()
    assert d['threads'][0]['hex'].upper() == '#000080'
    assert 38 <= d['report']['width_mm'] <= 42, d['report']

    # text-only SVG: a clear error, not "no drawable shapes"
    r = _import_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
                    '<text x="1" y="8">Hi</text></svg>')
    assert r.status_code == 400 and 'outlines' in r.json()['detail']
    # shapes plus text: stitched, with a warning about the skipped text
    d = _import_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50">'
                    '<rect x="5" y="5" width="30" height="30"/><text x="1" y="48">Hi</text></svg>').json()
    assert d['warnings'] and 'text' in d['warnings'][0]


def test_svg_colour_blocks_merge_when_layering_allows():
    from inkstitchlib import svginput
    # blue, cream, blue, cream ... per letter -> two blocks, not eight,
    # because the outline pieces never cover an earlier fill
    letters = ''.join('<path d="M%d,10 h8 v20 h-8 z" fill="#fdf0e1" stroke="#0047ab" '
                      'stroke-width="2" paint-order="stroke"/>' % x for x in range(10, 90, 20))
    pat, layers, info = svginput.digitize_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 40">%s</svg>' % letters)
    assert len(layers) == 2
    # a fill sewn over an earlier same-colour piece must not be merged under it
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 40">'
           '<rect x="10" y="5" width="30" height="30" fill="#0047ab"/>'
           '<rect x="20" y="10" width="30" height="20" fill="#fdf0e1"/>'
           '<rect x="30" y="15" width="30" height="10" fill="#0047ab"/></svg>')
    pat, layers, info = svginput.digitize_svg(svg)
    assert [L['hex'] for L in layers] == ['#0047AB', '#FDF0E1', '#0047AB']


def _phase_log(monkeypatch):
    """Record which kind of stitching each sew call is, in order."""
    from digitizer import core
    from inkstitchlib import fills
    log = []
    real_edge, real_area, real_col = core.sew_edge_run, fills.sew_area, core.sew_column
    monkeypatch.setattr(core, 'sew_edge_run', lambda *a, **k: (log.append('U'), real_edge(*a, **k)))
    monkeypatch.setattr(fills, 'sew_area', lambda *a, **k: (log.append('T'), real_area(*a, **k)))

    def col(s, c, travel, phase='both'):
        log.append('U' if phase == 'underlay' else 'T')
        return real_col(s, c, travel, phase)
    monkeypatch.setattr(core, 'sew_column', col)
    return log


def _underlay_first(log):
    # once the first top stitch of a block goes down, no more underlay follows
    return 'T' in log and log.index('T') > 0 and 'U' not in log[log.index('T'):]


def test_underlay_sewn_first_per_block(monkeypatch):
    from inkstitchlib import svginput, layers as veclayers
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 60">'
           + ''.join('<rect x="%d" y="10" width="14" height="30" fill="#1a3b69"/>' % x
                     for x in (5, 30, 55, 80)) + '</svg>')
    log = _phase_log(monkeypatch)
    pat, layers, info = svginput.digitize_svg(svg)
    assert log.count('U') >= 4 and log.count('T') >= 4 and _underlay_first(log), log

    # AI layers: four separate objects in one layer
    lyr = [{'name': 'Bars', 'color': '#1a3b69', 'visible': True,
            'params': {'stitch': 'fill'},
            'polys': [{'shell': [[x, 10], [x + 14, 10], [x + 14, 40], [x, 40]], 'holes': []}
                      for x in (5, 30, 55, 80)]}]
    log.clear()
    veclayers.stitch(lyr)
    assert log.count('U') >= 4 and log.count('T') >= 4 and _underlay_first(log), log


def test_image_digitizer_underlay_first(monkeypatch):
    from digitizer import segment, engine
    from PIL import Image as PImage, ImageDraw as PDraw
    im = PImage.new('RGBA', (400, 200), (0, 0, 0, 0))
    d = PDraw.Draw(im)
    for x in (20, 120, 220, 320):
        d.rectangle([x, 40, x + 60, 160], fill=(26, 59, 105, 255))
    p = os.path.join(tempfile.mkdtemp(), 'bars.png')
    im.save(p)
    log = _phase_log(monkeypatch)
    layers = segment.quantize(segment.load(p), 1)
    engine.build_pattern(layers, 80.0, 400, engine.Params(target_width_mm=80, heavy_underlay=True))
    assert log.count('U') >= 4 and log.count('T') >= 4 and _underlay_first(log), log


def test_hoops():
    r = client.get('/api/hoops').json()
    assert any(h['name'] == '5" × 7"' and h['w_mm'] == 130 for h in r)
    assert not any(h['custom'] for h in r)
    r = client.post('/api/hoops', json={'name': 'Cap left', 'w_mm': 60, 'h_mm': 40})
    assert r.status_code == 200 and r.json()['custom'] and r.json()['name'] == 'Cap left'
    hid = r.json()['id']
    assert any(h['id'] == hid for h in client.get('/api/hoops').json())
    assert client.post('/api/hoops', json={'w_mm': 5, 'h_mm': 40}).status_code == 400
    assert client.post('/api/hoops', json={'w_mm': 'x'}).status_code == 400
    assert client.delete('/api/hoops/%d' % hid).status_code == 200
    assert client.delete('/api/hoops/%d' % hid).status_code == 404


def test_presets_and_stitch_settings():
    r = client.get('/api/presets').json()
    assert any(f['id'] == 'fleece' and f['knockdown'] for f in r['fabrics'])
    assert any(m['id'] == 'tajima' and m['format'] == 'dst' for m in r['machines'])
    from app import _stitch_settings
    st = _stitch_settings({'settings': {'underlay': 'bogus', 'pull_comp': 5, 'knockdown': 1}})
    assert st == {'underlay': 'auto', 'pull_comp': 1.0, 'min_satin': 1.0,
                  'knockdown': True, 'cap_mode': False}


def test_knockdown_and_cap_mode_on_image(image_job):
    job, base = image_job
    r = client.post('/api/digitize', data={
        'job': job, 'colors': 2, 'width_mm': 60, 'hoop_w': 100, 'hoop_h': 100,
        'density': 0.4, 'max_satin': 8, 'autotune': False, 'underlay': 'heavy',
        'knockdown': True, 'cap_mode': True, 'pull_comp': 0.4, 'fabric': 'fleece'})
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d['threads']) == 3                      # knockdown + 2 colours
    assert d['report']['colour_changes'] == 2
    assert d['report']['stitches'] > base['report']['stitches']
    assert len(client.get('/api/stitches/%s' % job).json()) == 3
    from app import _load_meta
    assert _load_meta(job)['layers'][0]['name'] == 'Knockdown'


def test_pull_compensation_widens_satin():
    from inkstitchlib import svginput
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="60mm" height="30mm" viewBox="0 0 60 30">'
           '<path d="M5,10 h50 v6 h-50 z" fill="#1a3b69"/></svg>')
    narrow, _l, _i = svginput.digitize_svg(svg, pull_comp=0.0)
    wide, _l, _i = svginput.digitize_svg(svg, pull_comp=0.6)

    import pystitch

    def width(pat):
        xs = [x for x, y, c in pat.stitches if (c & 0xFF) == pystitch.STITCH]
        return (max(xs) - min(xs)) / 10.0
    assert width(narrow) < 50.3 and width(wide) > width(narrow) + 0.5


def test_narrow_satin_becomes_running_stitch():
    from inkstitchlib import pen
    thin = [{'mode': 'center', 'color': '#a8201a', 'width_mm': 0.6, 'spacing_mm': 0.4,
             'points': [[0, 0], [30, 0]]}]
    fat = [{'mode': 'center', 'color': '#a8201a', 'width_mm': 3.0, 'spacing_mm': 0.4,
            'points': [[0, 0], [30, 0]]}]
    pt, _ = pen.build(thin)
    pf, _ = pen.build(fat)
    import pystitch
    ys_thin = [abs(y) for x, y, c in pt.stitches if (c & 0xFF) == pystitch.STITCH]
    ys_fat = [abs(y) for x, y, c in pf.stitches if (c & 0xFF) == pystitch.STITCH]
    assert max(ys_thin) < 3 and max(ys_fat) > 12         # 0.1 mm units: run vs 3 mm satin


def test_layers_colour_sequencing_applique_and_puff():
    from inkstitchlib import layers as veclayers
    import pystitch
    box = lambda x, y, w, h: {'shell': [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], 'holes': []}
    blue1 = {'name': 'A', 'color': '#1a3b69', 'polys': [box(0, 0, 20, 20)], 'params': {'stitch': 'fill'}}
    red = {'name': 'B', 'color': '#a8201a', 'polys': [box(40, 0, 20, 20)], 'params': {'stitch': 'fill'}}
    blue2 = {'name': 'C', 'color': '#1a3b69', 'polys': [box(80, 0, 20, 20)], 'params': {'stitch': 'fill'}}
    pat, blocks = veclayers.stitch([blue1, red, blue2])
    assert [b['hex'] for b in blocks] == ['#1A3B69', '#A8201A']     # blue layers merged
    # red overlapping the first blue must keep a third block
    red_over = dict(red, polys=[box(10, 5, 20, 10)])
    pat, blocks = veclayers.stitch([blue1, red_over, dict(blue2, polys=[box(15, 8, 20, 4)])])
    assert [b['hex'] for b in blocks] == ['#1A3B69', '#A8201A', '#1A3B69']

    app_layer = {'name': 'Patch', 'color': '#1d7a4c', 'polys': [box(0, 0, 30, 30)],
                 'params': {'stitch': 'applique', 'border_mm': 2.0}}
    pat, blocks = veclayers.stitch([app_layer])
    assert pat.count_stitch_commands(pystitch.STOP) == 2 and len(blocks) == 1
    puff = {'name': 'Puff', 'color': '#f0b428', 'polys': [box(0, 0, 40, 6)],
            'params': {'stitch': 'puff'}}
    pat, blocks = veclayers.stitch([puff], settings={'knockdown': True})
    assert blocks[0]['name'] == 'Knockdown' and len(blocks) == 2
    r = client.post('/api/stitch_layers', json={'layers': [app_layer, puff],
                                                'settings': {'cap_mode': True}})
    assert r.status_code == 200, r.text


def test_sketch_mode():
    from PIL import Image as PImage, ImageDraw as PDraw
    im = PImage.new('RGB', (400, 300), (255, 255, 255))
    d = PDraw.Draw(im)
    d.ellipse([40, 40, 240, 240], outline=(30, 30, 30), width=6)
    d.line([260, 60, 380, 260], fill=(30, 30, 30), width=6)
    buf = io.BytesIO(); im.save(buf, 'PNG')
    r = client.post('/api/sketch', files={'image': ('photo.png', buf.getvalue(), 'image/png')},
                    data={'width_mm': 80, 'detail': 3, 'color': '#a8201a'})
    assert r.status_code == 200, r.text
    dj = r.json()
    assert dj['kind'] == 'sketch' and dj['sketch_info']['lines'] >= 2
    # the drawing spans 340 of the 400 px -> 68 mm of the 80 mm image width
    assert 64 <= dj['report']['width_mm'] <= 72 and len(dj['threads']) == 1
    single = client.post('/api/sketch', files={'image': ('photo.png', buf.getvalue(), 'image/png')},
                         data={'width_mm': 80, 'detail': 3, 'bean': False}).json()
    assert single['report']['stitches'] < dj['report']['stitches'] * 0.6
    blank = PImage.new('RGB', (200, 200), (255, 255, 255)); b2 = io.BytesIO(); blank.save(b2, 'PNG')
    assert client.post('/api/sketch', files={'image': ('b.png', b2.getvalue(), 'image/png')}).status_code == 400


def test_drawn_shapes_lines_and_fill_patterns():
    from inkstitchlib import layers as veclayers
    import pystitch
    box = lambda x, y, w, h: {'shell': [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], 'holes': []}
    line = {'points': [[0, 0], [30, 0], [30, 20]]}
    run = {'name': 'Run', 'color': '#a8201a', 'lines': [line], 'params': {'stitch': 'run', 'run_len_mm': 2.0}}
    bean = {'name': 'Bean', 'color': '#a8201a', 'lines': [line], 'params': {'stitch': 'bean', 'run_len_mm': 2.0}}
    satin = {'name': 'Satin', 'color': '#a8201a', 'lines': [line], 'params': {'stitch': 'satin', 'width_mm': 3.0, 'density': 0.4}}
    n = {}
    for L in (run, bean, satin):
        pat, blocks = veclayers.stitch([L])
        n[L['name']] = sum(1 for q in pat.stitches if (q[2] & 0xFF) == pystitch.STITCH)
        assert len(blocks) == 1
    assert n['Bean'] > n['Run'] * 2.0 and n['Satin'] > n['Bean']
    # satin has real width; run stays on the line
    pat, _ = veclayers.stitch([satin])
    ys = [y for x, y, c in pat.stitches if (c & 0xFF) == pystitch.STITCH and x < 100]
    assert max(ys) - min(ys) > 25
    # fill patterns: walk is much lighter than tatami; satin pattern sews a blob
    fill = {'name': 'F', 'color': '#1a3b69', 'polys': [box(0, 0, 20, 12)], 'params': {'stitch': 'fill', 'fill_method': 'tatami'}}
    walk = dict(fill, params={'stitch': 'fill', 'fill_method': 'walk'})
    sat = dict(fill, params={'stitch': 'fill', 'fill_method': 'satin'})
    cnt = lambda p: sum(1 for q in p.stitches if (q[2] & 0xFF) == pystitch.STITCH)
    assert cnt(veclayers.stitch([walk])[0]) < cnt(veclayers.stitch([fill])[0]) * 0.5
    assert cnt(veclayers.stitch([sat])[0]) > 100
    # mixed through the API, with a line and a polygon in one layer
    r = client.post('/api/stitch_layers', json={'layers': [dict(fill, lines=[line])]})
    assert r.status_code == 200, r.text
    r = client.post('/api/stitch_layers', json={'layers': [{'name': 'x', 'color': '#000', 'lines': [{'points': [[0, 0]]}]}]})
    assert r.status_code == 400


def test_thread_catalog():
    r = client.get('/api/threads/all')
    assert r.status_code == 200
    d = r.json()
    assert d['brands'] > 70 and d['colors'] > 15000 and len(d['threads']) == d['colors']
    t = d['threads'][0]
    assert set(t) == {'brand', 'name', 'number', 'hex'} and t['hex'].startswith('#')


def test_decorative_run_types():
    from inkstitchlib import layers as veclayers
    import pystitch
    line = {'points': [[0, 0], [40, 0]]}
    base = {'name': 'L', 'color': '#a8201a', 'lines': [line], 'params': {'stitch': 'run', 'run_len_mm': 2.5, 'width_mm': 3.0}}
    counts, spread = {}, {}
    for t in ('run', 'estitch', 'triangle', 'cross', 'motif'):
        pat, blocks = veclayers.stitch([dict(base, params=dict(base['params'], stitch=t))])
        st = [(x, y) for x, y, c in pat.stitches if (c & 0xFF) == pystitch.STITCH]
        counts[t] = len(st)
        ys = [y for x, y in st]
        spread[t] = (max(ys) - min(ys)) / 10.0
    for t in ('estitch', 'triangle', 'cross', 'motif'):
        assert counts[t] > counts['run'] * 1.4, (t, counts)
        assert 2.4 <= spread[t] <= 4.5, (t, spread)          # ~ the 3 mm width
    assert spread['run'] < 1.0
    # decorative outline on a closed shape, and through the API
    box = {'shell': [[0, 0], [30, 0], [30, 20], [0, 20]], 'holes': []}
    pat, blocks = veclayers.stitch([{'name': 'B', 'color': '#1a3b69', 'polys': [box], 'params': {'stitch': 'cross'}}])
    assert sum(1 for q in pat.stitches if (q[2] & 0xFF) == pystitch.STITCH) > 80
    r = client.post('/api/stitch_layers', json={'layers': [dict(base, params={'stitch': 'motif'})]})
    assert r.status_code == 200, r.text


def test_fill_patterns_and_previews():
    from inkstitchlib import layers as veclayers, patterns
    import pystitch
    sq = {'shell': [[0, 0], [16, 0], [16, 16], [0, 16]], 'holes': []}
    counts = {}
    for name in veclayers.FILL_METHODS:
        pat, blocks = veclayers.stitch([{'name': name, 'color': '#3A78B5', 'polys': [sq],
                                         'params': {'stitch': 'fill', 'fill_method': name, 'underlay': 'none'}}])
        counts[name] = sum(1 for q in pat.stitches if (q[2] & 0xFF) == pystitch.STITCH)
        assert counts[name] > 80, name
    # a patterned fill puts its needle points where the motif crosses the rows
    assert counts['diamonds_sm'] > counts['tatami'] and counts['hearts_md'] > counts['tatami']
    assert set(patterns.PATTERN_FILLS) <= set(veclayers.FILL_METHODS)
    r = client.get('/api/fills')
    assert r.status_code == 200 and [f['id'] for f in r.json()] == list(veclayers.FILL_METHODS)
    r = client.get('/api/fill_preview/hearts_md.png')
    assert r.status_code == 200 and r.headers['content-type'] == 'image/png' and r.content[:4] == b'\x89PNG'
    assert client.get('/api/fill_preview/nope.png').status_code == 404


def test_quote_v2_templates_and_export_options(image_job):
    from inkstitchlib import worksheet
    q = worksheet.quote(10000, setup=25, price_per_1000=2, min_digitizing=30, garment_qty=12,
                        garment_base=8, markup_pct=50, run_per_1000=1, colour_changes=3,
                        colour_fee=0.5, extra_per_piece=1, discount_pct=10, rush_pct=0, tax_pct=8)
    assert q['digitizing'] == 30                      # minimum beats 10k × $2/1000
    assert abs(q['marked_up'] - 144) < 1e-9 and abs(q['run_piece'] - 12.5) < 1e-9
    assert abs(q['run'] - 150) < 1e-9 and abs(q['discount'] - 29.4) < 1e-9
    assert abs(q['subtotal'] - (25 + 30 + 144 + 150 - 29.4)) < 1e-9
    assert abs(q['total'] - q['subtotal'] * 1.08) < 1e-9 and abs(q['per_piece'] - q['total'] / 12) < 1e-9
    r = client.post('/api/quote_calc', json={'stitches': 5000, 'colour_changes': 2, 'params': {'price_per_1000': 3, 'colour_fee': 1, 'garment_qty': 2}})
    assert r.status_code == 200 and abs(r.json()['digitizing'] - 15) < 1e-9 and abs(r.json()['run'] - 4) < 1e-9
    r = client.post('/api/quotes', json={'name': 'Left chest', 'params': {'setup': 20, 'price_per_1000': 2.5}})
    assert r.status_code == 200
    qid = r.json()['id']
    tpls = client.get('/api/quotes').json()
    assert any(t['id'] == qid and t['params']['setup'] == 20 for t in tpls)
    assert client.delete('/api/quotes/%d' % qid).status_code == 200
    job, _ = image_job
    r = client.get('/api/quote/%s.pdf?setup=20&price_per_1000=2&garment_qty=3&garment_base=5&name=Test' % job)
    assert r.status_code == 200 and r.content[:4] == b'%PDF'
    # a single format at a chosen origin, and a zip of several with both PDFs
    r = client.get('/api/download/%s?fmt=pes&origin=tl&name=My%%20Logo' % job)
    assert r.status_code == 200 and 'Logo.pes' in r.headers['content-disposition']
    import io, zipfile
    r = client.get('/api/download/%s?formats=dst,jef&worksheet_pdf=1&quote_pdf=1&setup=10&name=Logo' % job)
    assert r.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert 'Logo.dst' in names and 'Logo.jef' in names and 'Logo-worksheet.pdf' in names and 'Logo-quote.pdf' in names
    r = client.get('/api/download/%s?fmt=zip' % job)
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert 'design.dst' in names and 'design-worksheet.pdf' in names


def test_underlay_list_and_counts():
    from inkstitchlib import layers as veclayers
    import pystitch
    box = {'shell': [[0, 0], [20, 0], [20, 10], [0, 10]], 'holes': []}
    L = lambda prm: [{'id': 'a1', 'name': 'A', 'color': '#111111', 'polys': [box], 'params': dict({'stitch': 'fill'}, **prm)}]
    n = lambda prm: sum(1 for q in veclayers.stitch(L(prm))[0].stitches if (q[2] & 0xFF) == pystitch.STITCH)
    none = n({'underlays': []})
    one = n({'underlays': [{'type': 'contour', 'len_mm': 2}]})
    two = n({'underlays': [{'type': 'contour', 'len_mm': 2}, {'type': 'zigzag', 'len_mm': 3}]})
    assert none < one < two
    assert veclayers.STITCH_COUNTS.get('a1', 0) > 0
    r = client.post('/api/stitch_layers', json={'layers': L({'underlays': [{'type': 'center'}], 'short_frac': 0.3})})
    assert r.status_code == 200 and r.json()['counts']['a1'] > 50


def test_stitch_options_tie_trim_speed():
    from inkstitchlib import layers as veclayers
    import pystitch
    box = {'shell': [[0, 0], [20, 0], [20, 10], [0, 10]], 'holes': []}
    box2 = {'shell': [[30, 0], [50, 0], [50, 10], [30, 10]], 'holes': []}
    L = lambda prm: [{'name': 'A', 'color': '#111111', 'polys': [box], 'params': dict({'stitch': 'fill'}, **prm)},
                     {'name': 'B', 'color': '#111111', 'polys': [box2], 'params': {'stitch': 'fill'}}]
    cmds = lambda pat: [c & 0xFF for x, y, c in pat.stitches]
    base = cmds(veclayers.stitch(L({}))[0])
    trimmed = cmds(veclayers.stitch(L({'trim_after': True}))[0])
    assert trimmed.count(pystitch.TRIM) > base.count(pystitch.TRIM)
    no_ties = cmds(veclayers.stitch(L({'tie_on': False, 'tie_off': False}))[0])
    assert no_ties.count(pystitch.STITCH) < base.count(pystitch.STITCH)
    slow = cmds(veclayers.stitch(L({'speed': 'slow'}))[0])
    assert pystitch.SLOW in slow and pystitch.SLOW not in base


def test_gradient_fill():
    from inkstitchlib import layers as veclayers
    import pystitch
    import numpy as np
    box = {'shell': [[0, 0], [30, 0], [30, 30], [0, 30]], 'holes': []}
    F = lambda prm: {'name': 'F', 'color': '#1a3b69', 'polys': [box],
                     'params': dict({'stitch': 'fill', 'angle': 0, 'underlay': 'none'}, **prm)}
    st = lambda pat: np.array([(x, y) for x, y, c in pat.stitches if (c & 0xFF) == pystitch.STITCH], float) / 10.0
    plain = st(veclayers.stitch([F({})])[0])
    grad = st(veclayers.stitch([F({'gradient': True, 'gradient_to_mm': 2.0})])[0])
    flip = st(veclayers.stitch([F({'gradient': True, 'gradient_to_mm': 2.0, 'gradient_flip': True})])[0])
    assert len(grad) < len(plain) * 0.6 and abs(len(grad) - len(flip)) < len(grad) * 0.15
    # dense end vs open end: many more needle points in one half of the shape
    top = (grad[:, 1] < 0).sum(); bottom = (grad[:, 1] >= 0).sum()
    assert max(top, bottom) > 2.0 * min(top, bottom)
    ftop = (flip[:, 1] < 0).sum(); fbottom = (flip[:, 1] >= 0).sum()
    assert (top > bottom) != (ftop > fbottom)
    r = client.post('/api/stitch_layers', json={'layers': [F({'gradient': True, 'fill_method': 'waves'})]})
    assert r.status_code == 200, r.text


def test_object_refinements():
    """Split satin, hand stitch, per-object underlay and the origin the studio
    uses to overlay stitches on the shapes."""
    from inkstitchlib import layers as veclayers
    import pystitch
    import numpy as np
    line = {'points': [[0, 0], [40, 0], [40, 30]]}
    L = lambda prm: {'name': 'L', 'color': '#a8201a', 'lines': [line], 'params': dict({'stitch': 'satin', 'width_mm': 10}, **prm)}
    st = lambda pat: np.array([(x, y) for x, y, c in pat.stitches if (c & 0xFF) == pystitch.STITCH], float)
    longest = lambda pat: np.linalg.norm(np.diff(st(pat), axis=0), axis=1).max() / 10.0
    assert longest(veclayers.stitch([L({'split': True, 'split_max_mm': 5})])[0]) < 5.5
    assert longest(veclayers.stitch([L({'split': False})])[0]) > 9.5
    plain = st(veclayers.stitch([L({'width_mm': 3, 'hand': 0})])[0])
    hand = st(veclayers.stitch([L({'width_mm': 3, 'hand': 0.8})])[0])
    assert len(plain) == len(hand) and np.abs(plain - hand).max() > 1.0 and np.abs(plain - hand).max() < 6.0
    box = {'shell': [[0, 0], [30, 0], [30, 20], [0, 20]], 'holes': []}
    F = lambda prm: {'name': 'F', 'color': '#1a3b69', 'polys': [box], 'params': dict({'stitch': 'fill'}, **prm)}
    n = lambda prm: len(st(veclayers.stitch([F(prm)])[0]))
    assert n({'underlay': 'none'}) < n({'underlay': 'light'}) < n({'underlay': 'heavy'})
    # row shortening moves inner needle points on a tight ring; the count stays
    ring = {'shell': [[10 + 5 * np.cos(t), 10 + 5 * np.sin(t)] for t in np.linspace(0, 2 * np.pi, 60, endpoint=False)], 'holes': []}
    R = lambda prm: {'name': 'R', 'color': '#000', 'polys': [ring], 'params': dict({'stitch': 'outline', 'border_mm': 3.0}, **prm)}
    a, b = st(veclayers.stitch([R({'row_short': True})])[0]), st(veclayers.stitch([R({'row_short': False})])[0])
    # same stitch budget (± the closing travel), inner needle points moved out
    assert abs(len(a) - len(b)) < 12 and len(a) > 100
    n = min(len(a), len(b))
    assert np.abs(a[:n] - b[:n]).max() > 3.0      # 0.1 mm units
    r = client.post('/api/stitch_layers', json={'layers': [F({'fill_method': 'waves', 'stitch_len_mm': 5, 'underpath': False, 'pull_comp_mm': 0.4})]})
    assert r.status_code == 200, r.text
    o = r.json()['origin_mm']
    assert 14 < o[0] < 16 and 9 < o[1] < 11
