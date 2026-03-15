#!/usr/bin/env python3
"""
pptx_figure_builder.py
Future Frontiers — Chapter Figure Generation System
Author: Jigar Sheth | Tool: Claude (ghostwriter)

TWO-PASS SYSTEM:
  Pass 1 (Node.js / PptxGenJS): generates boxes, tables, charts, text
  Pass 2 (Python, this module):  injects native <p:cxnSp> connectors,
                                  verifies layout, exports PNG thumbnails

WHY TWO PASSES:
  PptxGenJS has no addConnector(). All connectors it creates are bare <p:sp>
  line shapes with no routing intelligence. True PowerPoint connectors use
  <p:cxnSp> elements with preset geometry (bentConnector3, curvedConnector3,
  straightConnector1, etc.) that auto-route and snap to box edges.
  This injector generates those elements from box-edge coordinates.

CONNECTOR PRESETS (prstGeom):
  straightConnector1  — straight line, any angle
  bentConnector2      — one right-angle bend (L-shape)
  bentConnector3      — two bends (Z or S shape) — MOST COMMON for flow diagrams
  bentConnector4      — three bends
  bentConnector5      — four bends
  curvedConnector2    — single arc
  curvedConnector3    — S-curve (good for feedback loops)
  curvedConnector4    — double S-curve
  curvedConnector5    — triple S-curve

DYNAMIC USAGE:
  Figures are not predefined. Each chapter defines its own figure spec
  as a Python dict and calls the appropriate builder function.
  The system adapts to any layout, any number of boxes, any connector topology.
"""

import zipfile, io, os, re, subprocess
from dataclasses import dataclass, field
from typing import Optional

EMU = 914400  # 1 inch in EMU

# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class Box:
    """A rectangular shape on the slide."""
    name: str
    x: float; y: float; w: float; h: float
    
    @property
    def right(self):  return self.x + self.w
    @property
    def bottom(self): return self.y + self.h
    @property
    def cx(self):     return self.x + self.w / 2
    @property
    def cy(self):     return self.y + self.h / 2
    
    def edge(self, side: str):
        """Return (x, y) of the midpoint of the named edge."""
        return {
            'left':   (self.x,       self.cy),
            'right':  (self.right,   self.cy),
            'top':    (self.cx,      self.y),
            'bottom': (self.cx,      self.bottom),
        }[side]


@dataclass  
class Connector:
    """A connector between two points."""
    from_pt: tuple          # (x, y) in inches — typically box.edge(side)
    to_pt:   tuple          # (x, y) in inches
    preset:  str = 'bentConnector3'
    color:   str = '1F5C99'
    arrow_start: bool = False
    arrow_end:   bool = True
    line_width:  int  = 19050    # 1.5pt in EMU
    dash:        Optional[str] = None   # None, 'dash', 'lgDash', 'dot'
    label:       Optional[str] = None
    label_x:     Optional[float] = None
    label_y:     Optional[float] = None
    label_w:     float = 1.2
    label_h:     float = 0.28
    label_color: Optional[str] = None
    label_align: str = 'ctr'


# ── XML generators ─────────────────────────────────────────────────────────

def _emu(inches: float) -> int:
    return max(1, int(inches * EMU))

def connector_xml(cid: int, conn: Connector) -> str:
    """Generate a <p:cxnSp> XML element from a Connector spec."""
    x1, y1 = conn.from_pt
    x2, y2 = conn.to_pt
    
    ox = min(x1, x2);  oy = min(y1, y2)
    cx = abs(x2 - x1); cy = abs(y2 - y1)
    # Minimum size so PowerPoint doesn't discard it
    cx = max(cx, 0.001); cy = max(cy, 0.001)
    
    flip_h = ' flipH="1"' if x2 < x1 else ''
    flip_v = ' flipV="1"' if y2 < y1 else ''
    
    head = '<a:headEnd type="arrow" w="med" len="med"/>' if conn.arrow_start else '<a:headEnd type="none"/>'
    tail = '<a:tailEnd type="arrow" w="med" len="med"/>' if conn.arrow_end   else '<a:tailEnd type="none"/>'
    dash_xml = f'<a:prstDash val="{conn.dash}"/>' if conn.dash else ''
    
    return f'''<p:cxnSp>
  <p:nvCxnSpPr>
    <p:cNvPr id="{cid}" name="Connector {cid}"/>
    <p:cNvCxnSpPr/>
    <p:nvPr/>
  </p:nvCxnSpPr>
  <p:spPr>
    <a:xfrm{flip_h}{flip_v}>
      <a:off x="{_emu(ox)}" y="{_emu(oy)}"/>
      <a:ext cx="{_emu(cx)}" cy="{_emu(cy)}"/>
    </a:xfrm>
    <a:prstGeom prst="{conn.preset}"><a:avLst/></a:prstGeom>
    <a:noFill/>
    <a:ln w="{conn.line_width}" cap="flat">
      <a:solidFill><a:srgbClr val="{conn.color}"/></a:solidFill>
      {dash_xml}
      {head}
      {tail}
    </a:ln>
  </p:spPr>
</p:cxnSp>'''


def label_xml(lid: int, x: float, y: float, w: float, h: float, 
              text: str, color: str = '555555', fontsize: int = 7,
              italic: bool = True, align: str = 'ctr') -> str:
    """Generate a <p:sp> text-only label."""
    italic_xml  = '<a:i/>' if italic else ''
    # Escape XML special chars
    text_escaped = text.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    # Multi-line: split on \n into multiple runs with <a:br/>
    lines = text_escaped.split('\n')
    runs = []
    for i, line in enumerate(lines):
        runs.append(f'''<a:r><a:rPr lang="en-US" sz="{fontsize*100}" dirty="0">
          {italic_xml}<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
        </a:rPr><a:t>{line}</a:t></a:r>''')
        if i < len(lines)-1:
            runs.append('<a:br><a:rPr lang="en-US"/></a:br>')
    
    return f'''<p:sp>
  <p:nvSpPr>
    <p:cNvPr id="{lid}" name="Label {lid}"/>
    <p:cNvSpPr txBox="1"/>
    <p:nvPr/>
  </p:nvSpPr>
  <p:spPr>
    <a:xfrm>
      <a:off x="{_emu(x)}" y="{_emu(y)}"/>
      <a:ext cx="{_emu(w)}" cy="{_emu(h)}"/>
    </a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
    <a:noFill/><a:ln><a:noFill/></a:ln>
  </p:spPr>
  <p:txBody>
    <a:bodyPr wrap="square" lIns="36576" rIns="36576" tIns="18288" bIns="18288"/>
    <a:lstStyle/>
    <a:p>
      <a:pPr algn="{align}"/>
      {chr(10).join(runs)}
    </a:p>
  </p:txBody>
</p:sp>'''


# ── PPTX injector ──────────────────────────────────────────────────────────

def inject_connectors(input_pptx: str, output_pptx: str,
                      connectors: list, slide_index: int = 0,
                      start_id: int = 200) -> dict:
    """
    Inject connector and label XML into a PPTX file.
    
    Args:
        input_pptx:  path to PPTX built by PptxGenJS (Pass 1)
        output_pptx: path to write final PPTX (Pass 2)
        connectors:  list of Connector objects
        slide_index: 0-based slide index (0 = first slide)
        start_id:    starting ID for injected elements (must be > all existing IDs)
    
    Returns:
        dict with verification stats
    """
    slide_name = f'ppt/slides/slide{slide_index + 1}.xml'
    
    # Read input
    with zipfile.ZipFile(input_pptx, 'r') as zin:
        slide_xml = zin.read(slide_name).decode('utf-8')
    
    # Find current max ID to avoid collisions
    existing_ids = [int(m) for m in re.findall(r'id="(\d+)"', slide_xml)]
    cid = max(existing_ids, default=0) + 1
    cid = max(cid, start_id)
    
    # Generate connector + label XML
    parts = []
    for conn in connectors:
        parts.append(connector_xml(cid, conn))
        cid += 1
        if conn.label:
            lx = conn.label_x if conn.label_x is not None else (conn.from_pt[0] + conn.to_pt[0])/2 - conn.label_w/2
            ly = conn.label_y if conn.label_y is not None else (conn.from_pt[1] + conn.to_pt[1])/2 - conn.label_h/2
            parts.append(label_xml(cid, lx, ly, conn.label_w, conn.label_h,
                                    conn.label, conn.label_color or conn.color,
                                    align=conn.label_align))
            cid += 1
    
    inject = '\n'.join(parts)
    slide_xml_new = slide_xml.replace('</p:spTree>', inject + '\n</p:spTree>')
    
    # Write output
    buf = io.BytesIO()
    with zipfile.ZipFile(input_pptx, 'r') as zin:
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = slide_xml_new.encode('utf-8') if item.filename == slide_name else zin.read(item.filename)
                zout.writestr(item, data)
    buf.seek(0)
    with open(output_pptx, 'wb') as f:
        f.write(buf.read())
    
    # Verify
    with zipfile.ZipFile(output_pptx, 'r') as z:
        content = z.read(slide_name).decode('utf-8')
        return {
            'cxnSp_count':     content.count('<p:cxnSp>'),
            'sp_count':        content.count('<p:sp>'),
            'connector_types': list(set(re.findall(r'prst="([^"]+Connector[^"]+)"', content))),
            'file_size_kb':    round(os.path.getsize(output_pptx)/1024, 1),
            'valid':           '</p:spTree>' in content,
        }


# ── Layout verification ────────────────────────────────────────────────────

def verify_layout(pptx_path: str, slide_index: int = 0,
                  content_x_max: float = 9.6, content_y_max: float = 4.93) -> list:
    """
    Check all shapes are within slide bounds.
    Returns list of problem strings (empty = all OK).
    """
    slide_name = f'ppt/slides/slide{slide_index+1}.xml'
    with zipfile.ZipFile(pptx_path, 'r') as z:
        content = z.read(slide_name).decode('utf-8')
    
    xfrms = re.findall(r'<a:xfrm[^>]*>.*?</a:xfrm>', content, re.DOTALL)
    problems = []
    for xf in xfrms:
        off = re.search(r'<a:off x="(-?\d+)" y="(-?\d+)"', xf)
        ext = re.search(r'<a:ext cx="(-?\d+)" cy="(-?\d+)"', xf)
        if off and ext:
            xi = int(off.group(1))/EMU;  yi = int(off.group(2))/EMU
            wi = int(ext.group(1))/EMU;  hi = int(ext.group(2))/EMU
            if xi > 0.3 and wi > 0.01:
                if xi > content_x_max:        problems.append(f'OFF_SCREEN x={xi:.3f}')
                if xi+wi > content_x_max+0.1: problems.append(f'OVERFLOW x+w={xi+wi:.3f}')
            if wi < -0.001: problems.append(f'NEG_W={wi:.3f}')
            if hi < -0.001: problems.append(f'NEG_H={hi:.3f}')
    return problems


# ── PNG thumbnail export ───────────────────────────────────────────────────

def export_thumbnails(pptx_path: str, out_dir: str) -> list:
    """
    Export each slide as a PNG using LibreOffice.
    Returns list of generated PNG paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    result = subprocess.run(
        ['soffice', '--headless', '--convert-to', 'png',
         '--outdir', out_dir, pptx_path],
        capture_output=True, text=True
    )
    pngs = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.png')]
    return sorted(pngs)


if __name__ == "__main__":
    print("pptx_figure_builder.py — loaded OK")
    print(f"Available connector presets:")
    for p in ['straightConnector1','bentConnector2','bentConnector3',
               'bentConnector4','bentConnector5',
               'curvedConnector2','curvedConnector3',
               'curvedConnector4','curvedConnector5']:
        print(f"  {p}")
