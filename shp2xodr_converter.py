import geopandas as gpd
import math, os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import defaultdict

BASE     = os.path.dirname(os.path.abspath(__file__))
SHP_DIR  = os.path.join(BASE, "shp_road", "Daegu")
# SHP_DIR  = os.path.join(BASE, "shp_road", "Pangyo_Zerocity")
# SHP_DIR  = os.path.join(BASE, "shp_road", "SangAm")
OUT_PATH = os.path.join(BASE, "output_A1A2B2.xodr")

EXCLUDE_LINK   = {'1'}
LANE_WIDTH     = 3.5
STRAIGHT_THR   = 0.5
RDP_EPSILON    = 0.3
PARALLEL_DIST  = 1.0
TAPER_LEN      = 20.0
OVERLAP_THRESH = 0.8
WIDTH_MIN      = 1.0
WIDTH_MAX      = 5.0
GEO_REF        = ("+proj=tmerc +lat_0=38 +lon_0=127 +k=1 "
                  "+x_0=200000 +y_0=600000 +ellps=GRS80 +units=m +no_defs")


def get_coords(geom):
    return [(p[0], p[1]) for p in geom.coords]

def dist2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _safe_int(val, default=0):
    try:    return int(str(val or default))
    except: return default

def _safe_str(val):
    s = str(val or '').strip()
    return '' if s.lower() == 'nan' else s

def _coords_length(pts):
    return sum(dist2(pts[i], pts[i+1]) for i in range(len(pts)-1))

def _taper_poly(W, L, opening):
    L = max(L, 1.0)
    c = 3*W / L**2
    d = -2*W / L**3
    return (0, 0, c, d) if opening else (W, 0, -c, -d)

def _perp_dists(coords):
    x0, y0 = coords[0]; x1, y1 = coords[-1]
    L = math.hypot(x1-x0, y1-y0)
    if L < 1e-4: return []
    return [abs((y1-y0)*x - (x1-x0)*y + x1*y0 - y1*x0) / L for x, y in coords[1:-1]]

def is_straight(coords):
    if len(coords) <= 2: return True
    d = _perp_dists(coords)
    return not d or (sum(d) / len(d)) <= STRAIGHT_THR

def rdp(coords, epsilon=RDP_EPSILON):
    if len(coords) <= 2: return coords
    dists = _perp_dists(coords)
    if not dists: return [coords[0], coords[-1]]
    idx = dists.index(max(dists))
    if dists[idx] > epsilon: 
        return rdp(coords[:idx+2], epsilon)[:-1] + rdp(coords[idx+1:], epsilon)
    return [coords[0], coords[-1]]

def catmull_splines(pts):
    T = []
    for i in range(len(pts)):
        if i == 0:            t = (pts[1][0]-pts[0][0], pts[1][1]-pts[0][1])
        elif i == len(pts)-1: t = (pts[-1][0]-pts[-2][0], pts[-1][1]-pts[-2][1])
        else:                 t = ((pts[i+1][0]-pts[i-1][0])/2, (pts[i+1][1]-pts[i-1][1])/2)
        T.append(t)
    return T

def build_planview(road_el, coords, ox, oy):
    if is_straight(coords):
        coords = [coords[0], coords[-1]]
    else:
        coords = rdp(coords)
    pts = [(x-ox, y-oy) for x, y in coords]
    pv  = ET.SubElement(road_el, "planView")
    s   = 0.0

    if len(pts) == 2:
        dx, dy = pts[1][0]-pts[0][0], pts[1][1]-pts[0][1]
        L = math.hypot(dx, dy)
        if L < 1e-4:
            return 0.0
        g = ET.SubElement(pv, "geometry",
                s="0", x=f"{pts[0][0]:.6e}", y=f"{pts[0][1]:.6e}",
                hdg=f"{math.atan2(dy, dx):.6e}", length=f"{L:.6e}")
        ET.SubElement(g, "line")
        return L

    T = catmull_splines(pts)
    for i in range(len(pts)-1):
        p0, p1 = pts[i], pts[i+1]
        dx, dy = p1[0]-p0[0], p1[1]-p0[1]
        L = math.hypot(dx, dy)
        if L < 1e-4: continue
        hdg = math.atan2(dy, dx)
        c, ss = math.cos(hdg), math.sin(hdg)
        u1  =  dx*c  + dy*ss;  v1  = -dx*ss  + dy*c
        tu0 =  T[i][0]*c   + T[i][1]*ss;   tv0 = -T[i][0]*ss   + T[i][1]*c
        tu1 =  T[i+1][0]*c + T[i+1][1]*ss; tv1 = -T[i+1][0]*ss + T[i+1][1]*c
        g = ET.SubElement(pv, "geometry",
                s=f"{s:.6e}", x=f"{p0[0]:.6e}", y=f"{p0[1]:.6e}",
                hdg=f"{hdg:.6e}", length=f"{L:.6e}")
        ET.SubElement(g, "paramPoly3",
                aU="0", bU=f"{tu0:.6e}", cU=f"{3*u1-2*tu0-tu1:.6e}", dU=f"{-2*u1+tu0+tu1:.6e}",
                aV="0", bV=f"{tv0:.6e}", cV=f"{3*v1-2*tv0-tv1:.6e}", dV=f"{-2*v1+tv0+tv1:.6e}",
                pRange="normalized")
        s += L
    return s

def build_a2_index(a2):
    a2f = a2[~a2['linktype'].astype(str).isin(EXCLUDE_LINK)]
    by_id        = {str(r['id']): r for _, r in a2f.iterrows()}
    rev_91_rid   = {}
    rev_91_node  = defaultdict(list)
    by_l_linkid  = defaultdict(list)
    by_from_node = defaultdict(list)
    for _, r in a2f.iterrows():
        fn = _safe_str(r.get('fromnodeid'))
        if fn:
            by_from_node[fn].append(r)
        if _safe_int(r.get('laneno')) >= 91:
            r_id = _safe_str(r.get('r_linkid'))
            if r_id:
                rev_91_rid[r_id] = r
            nk = (_safe_str(r.get('fromnodeid')), _safe_str(r.get('tonodeid')))
            rev_91_node[nk].append(r)
        l_id = _safe_str(r.get('l_linkid'))
        if l_id:
            by_l_linkid[l_id].append(r)
    return by_id, rev_91_rid, rev_91_node, by_l_linkid, by_from_node

def build_b2_maps(b2, a2_by_id):
    from shapely.geometry import Point

    def _chain(segs, geom):
        if not segs: return []
        if geom is None: return segs[0]
        v_sorted = sorted(segs, key=lambda seg: geom.project(Point(seg[len(seg)//2])))
        chained = []
        for seg in v_sorted:
            if not chained:
                a2_start = get_coords(geom)[0]
                if dist2(a2_start, seg[-1]) < dist2(a2_start, seg[0]):
                    seg = list(reversed(seg))
                chained.extend(seg)
            else:
                if dist2(chained[-1], seg[-1]) < dist2(chained[-1], seg[0]):
                    seg = list(reversed(seg))
                chained.extend(seg[1:] if dist2(chained[-1], seg[0]) < 1.0 else seg)
        return chained

    a2_geoms = {}
    for k, r in a2_by_id.items():
        try:    a2_geoms[k] = r['geometry']
        except: pass

    r_groups, l_groups = defaultdict(list), defaultdict(list)
    for _, row in b2.iterrows():
        coords = get_coords(row.geometry)
        r_id = _safe_str(row.get('r_linkid', ''))
        l_id = _safe_str(row.get('l_linkid', ''))
        if r_id: r_groups[r_id].append(coords)
        if l_id: l_groups[l_id].append(coords)

    b2_r = {k: _chain(v, a2_geoms.get(k)) for k, v in r_groups.items()}
    b2_l = {k: _chain(v, a2_geoms.get(k)) for k, v in l_groups.items()}
    return b2_r, b2_l

def resolve_coords(chain, b2_r, a2m_ids):
    coords = []
    for lk in chain:
        lid = str(lk['id'])
        if lid not in a2m_ids:
            continue
        pts = b2_r.get(lid)
        if pts is None:
            continue
        coords.extend(pts if not coords else pts[1:])
    return coords

def _b2_vertex_width(left_coords, right_coords):
    from shapely.geometry import LineString, Point
    if not left_coords or not right_coords or len(left_coords) < 2 or len(right_coords) < 2:

        return None
    right_line = LineString(right_coords)
    valid = [d for pt in left_coords
             if WIDTH_MIN < (d := Point(pt).distance(right_line)) < WIDTH_MAX]
    return round(sum(valid) / len(valid), 2) if valid else None

def _build_node_to_s(chain, b2_map):
    node_to_s = {}
    s = 0.0
    for lk in chain:
        node_to_s[_safe_str(lk.get('fromnodeid'))] = s
        pts = b2_map.get(str(lk['id']))
        if pts:
            s += _coords_length(pts)
    if chain:
        node_to_s[_safe_str(chain[-1].get('tonodeid'))] = s
    return node_to_s, s

def _walk_pocket_chain(start_lk, by_from_node, chain_nodes, chain_ids, visited):
    """포켓 시작 링크부터 ToNodeID가 main chain에 나타날 때까지 링크를 순서대로 반환"""
    result = []
    cur = start_lk
    for _ in range(50):  # 무한루프 방지
        cid = str(cur['id'])
        if cid in visited:
            break
        visited.add(cid)
        result.append(cur)
        to_node = _safe_str(cur.get('tonodeid'))
        if to_node in chain_nodes:
            break
        nxt_candidates = [r for r in by_from_node.get(to_node, [])
                          if str(r['id']) not in chain_ids
                          and str(r['id']) not in visited
                          and _safe_int(r.get('laneno')) == _safe_int(start_lk.get('laneno'))]
        if not nxt_candidates:
            break
        cur = nxt_candidates[0]
    return result

def _find_pocket_spans(chain, a2_by_id, by_from_node, by_l_linkid, b2_r, b2_l, node_to_s):
    """
    좌/우 포켓 차로 span 탐지.
    반환: [{'side': 'left'|'right', 's_start': float, 's_end': float,
             'stable_w': float, 'taper_open': float, 'taper_close': float}]
    """
    chain_ids   = {str(lk['id']) for lk in chain}
    chain_nodes = set(node_to_s.keys())
    visited     = set(chain_ids)
    spans       = []

    for lk in chain:
        from_node = _safe_str(lk.get('fromnodeid'))

        # 왼쪽 포켓 (LaneNo=91): from_node에서 시작하는 91 링크
        for r in by_from_node.get(from_node, []):
            if _safe_int(r.get('laneno')) < 91:
                continue
            pocket_chain = _walk_pocket_chain(r, by_from_node, chain_nodes, chain_ids, visited)
            if not pocket_chain:
                continue
            _register_pocket_span(pocket_chain, node_to_s, b2_r, b2_l, 'left', spans)

        # 오른쪽 포켓: chain 링크의 R_LinkID 체인 중 하나를 L_LinkID로 가리키며
        # 같은 from_node에서 시작하는 비체인 링크
        cur = lk
        while True:
            cur_id = str(cur['id'])
            for candidate in by_l_linkid.get(cur_id, []):
                cid = str(candidate['id'])
                if cid in visited:
                    continue
                if _safe_int(candidate.get('laneno')) >= 91:
                    continue
                if _safe_str(candidate.get('fromnodeid')) != from_node:
                    continue
                pocket_chain = _walk_pocket_chain(candidate, by_from_node, chain_nodes, chain_ids, visited)
                if not pocket_chain:
                    continue
                _register_pocket_span(pocket_chain, node_to_s, b2_r, b2_l, 'right', spans)
            r_id = _safe_str(cur.get('r_linkid'))
            if not r_id:
                break
            nxt = a2_by_id.get(r_id)
            if nxt is None or _safe_int(nxt.get('laneno')) >= 91:
                break
            cur = nxt

    return spans

def _register_pocket_span(pocket_chain, node_to_s, b2_r, b2_l, side, spans):
    s_start = node_to_s.get(_safe_str(pocket_chain[0].get('fromnodeid')))
    s_end   = node_to_s.get(_safe_str(pocket_chain[-1].get('tonodeid')))
    if s_start is None or s_end is None or s_end <= s_start:
        return

    widths = []
    for lk in pocket_chain:
        lid = str(lk['id'])
        w = _b2_vertex_width(b2_r.get(lid), b2_l.get(lid))
        widths.append(w)

    valid_ws = [w for w in widths if w is not None]
    if not valid_ws:
        return
    max_w = max(valid_ws)
    threshold = max_w * 0.3
    stable_ws = [w for w in valid_ws if w >= threshold]
    stable_w  = round(sum(stable_ws) / len(stable_ws), 2)

    # 링크별 누적 s로 taper 구간 길이 추정
    s = s_start
    taper_open = taper_close = 0.0
    in_open, in_close = True, False
    link_ss = [s_start]
    cur_s = s_start
    for lk in pocket_chain:
        pts = b2_r.get(str(lk['id'])) or b2_l.get(str(lk['id']))
        if pts and len(pts) >= 2:
            cur_s += _coords_length(pts)
        link_ss.append(cur_s)

    for i, lk in enumerate(pocket_chain):
        w = widths[i]
        seg_len = link_ss[i+1] - link_ss[i]
        if in_open:
            if w is None or w < threshold:
                taper_open += seg_len
            else:
                in_open = False
        if not in_open and not in_close:
            if w is None or w < threshold:
                in_close = True
        if in_close:
            taper_close += seg_len

    spans.append({
        'side': side, 's_start': s_start, 's_end': s_end,
        'stable_w': stable_w,
        'taper_open': max(taper_open, 1.0),
        'taper_close': max(taper_close, 1.0),
    })

def compute_chain_widths(chain, a2_by_id, rev_91_rid, b2_r, b2_l):
    lane_samples   = defaultdict(list)
    pocket_samples = []

    for lk in chain:
        cur, k = lk, 1
        while True:
            a2_id = str(cur['id'])
            w = _b2_vertex_width(b2_r.get(a2_id), b2_l.get(a2_id))
            if w:
                lane_samples[k].append(w)
            r_id = _safe_str(cur.get('r_linkid'))
            if not r_id: break
            nxt = a2_by_id.get(r_id)
            if nxt is None or _safe_int(nxt.get('laneno')) >= 91: break
            cur, k = nxt, k + 1

        pocket = rev_91_rid.get(str(lk['id']))
        if pocket is not None:
            w = _b2_vertex_width(b2_r.get(str(pocket['id'])), b2_l.get(str(pocket['id'])))
            if w:
                pocket_samples.append(w)

    result = {k: round(sum(ws)/len(ws), 2) for k, ws in lane_samples.items()}
    result['pocket'] = round(sum(pocket_samples)/len(pocket_samples), 2) \
                       if pocket_samples else LANE_WIDTH
    return result

def _link_lane_info(lk, a2_by_id, rev_91_rid, rev_91_node):
    cur = lk
    while True:
        r_id = _safe_str(cur.get('r_linkid'))
        if not r_id: break
        nxt = a2_by_id.get(r_id)
        if nxt is None or _safe_int(nxt.get('laneno')) >= 91: break
        cur = nxt
    n_right = max(1, _safe_int(cur.get('laneno'), 1))

    lk_id = str(lk['id'])
    start_91 = rev_91_rid.get(lk_id)
    if start_91 is None:
        l_id = _safe_str(lk.get('l_linkid'))
        if l_id:
            nxt = a2_by_id.get(l_id)
            if nxt is not None and _safe_int(nxt.get('laneno')) >= 91:
                start_91 = nxt
    if start_91 is None:
        nk = (_safe_str(lk.get('fromnodeid')), _safe_str(lk.get('tonodeid')))
        candidates = rev_91_node.get(nk, [])
        if candidates:
            start_91 = min(candidates, key=lambda r: _safe_int(r.get('laneno')))

    n_left = 0
    if start_91 is not None:
        cur = start_91
        while True:
            nxt = rev_91_rid.get(str(cur['id']))
            if nxt is None: break
            cur = nxt
        n_left = max(1, _safe_int(cur.get('laneno'), 91) - 90)

    return n_right, n_left

def _add_lane(parent, lane_id, pred_id=None, succ_id=None, width=None, mark_type="solid",
              poly=None, L_taper=None, sec_len=None):
    w = width if width is not None else LANE_WIDTH
    lane = ET.SubElement(parent, "lane", id=str(lane_id), type="driving", level="false")
    if pred_id is not None or succ_id is not None:
        lk_el = ET.SubElement(lane, "link")
        if pred_id is not None:
            ET.SubElement(lk_el, "predecessor", id=str(pred_id))
        if succ_id is not None:
            ET.SubElement(lk_el, "successor", id=str(succ_id))
    if poly and L_taper:
        a, b, c, d = poly
        opening = abs(a) < 1e-6
        ET.SubElement(lane, "width", sOffset="0",
                      a=f"{a:.6e}", b=f"{b:.6e}", c=f"{c:.6e}", d=f"{d:.6e}")
        final_w = w if opening else 0.0
        ET.SubElement(lane, "width", sOffset=f"{L_taper:.6e}",
                      a=f"{final_w:.6e}", b="0", c="0", d="0")
        # 개구+폐구 동시 처리
        if opening and sec_len is not None and sec_len > L_taper * 2 + 1.0:
            close_s = sec_len - L_taper
            ca, cb, cc, cd = _taper_poly(w, L_taper, False)
            ET.SubElement(lane, "width", sOffset=f"{close_s:.6e}",
                          a=f"{ca:.6e}", b=f"{cb:.6e}", c=f"{cc:.6e}", d=f"{cd:.6e}")
            ET.SubElement(lane, "width", sOffset=f"{sec_len:.6e}",
                          a="0", b="0", c="0", d="0")
    else:
        ET.SubElement(lane, "width", sOffset="0", a=f"{w:.6e}", b="0", c="0", d="0")
    ET.SubElement(lane, "roadMark", sOffset="0", type=mark_type, weight="standard",
                  color="standard", width="0.15")

def _add_laneSection(lanes_el, s, n_right, n_left, bilateral,
                     n_left_prev, n_left_next, chain_widths,
                     n_opposite=None, opposite_widths=None,
                     n_right_prev=None, n_right_next=None, L_taper=TAPER_LEN, sec_len=None,
                     r_pocket=False, r_pocket_prev=False, r_pocket_next=False,
                     right_stable_w=None, right_L_open=TAPER_LEN, right_L_close=TAPER_LEN):
    ls = ET.SubElement(lanes_el, "laneSection", s=f"{s:.6e}")
    cl = ET.SubElement(ET.SubElement(ls, "center"), "lane", id="0", type="none", level="false")
    ET.SubElement(cl, "roadMark", sOffset="0", type="none")

    if bilateral:
        n_left_cnt = n_opposite if n_opposite is not None else n_right
        opp_w = opposite_widths or chain_widths
        left = ET.SubElement(ls, "left")
        for i in range(1, n_left_cnt + 1):
            _add_lane(left, i, width=opp_w.get(i, LANE_WIDTH),
                      mark_type="none" if i == 1 else "solid")
        right = ET.SubElement(ls, "right")
        for i in range(1, n_right + 1):
            w = chain_widths.get(i, LANE_WIDTH)
            poly = lane_sec_len = None
            if i == n_right:
                opening = n_right_prev is not None and n_right > n_right_prev
                closing = n_right_next is not None and n_right > n_right_next
                if opening:
                    poly = _taper_poly(w, L_taper, True)
                    lane_sec_len = sec_len if closing else None
                elif closing:
                    poly = _taper_poly(w, L_taper, False)
            _add_lane(right, -i, width=w, poly=poly,
                      L_taper=L_taper if poly else None, sec_len=lane_sec_len)
        return

    right = ET.SubElement(ls, "right")
    for i in range(1, n_left + n_right + 1):
        pred_id = succ_id = poly = None
        lane_sec_len = None
        if i <= n_left:
            if n_left_prev is not None:
                pred_id = -i if i <= n_left_prev else None
            if n_left_next is not None:
                succ_id = -i if i <= n_left_next else None
            w = chain_widths.get('pocket', LANE_WIDTH)
            if i == n_left:
                if n_left_prev is not None and n_left > n_left_prev:
                    poly = _taper_poly(w, L_taper, True)
                elif n_left_next is not None and n_left > n_left_next:
                    poly = _taper_poly(w, L_taper, False)
            _add_lane(right, -i, pred_id, succ_id, width=w,
                      mark_type="none" if i == 1 else "solid",
                      poly=poly, L_taper=L_taper if poly else None)
        else:
            k = i - n_left
            if n_left_prev is not None:
                pred_id = -(n_left_prev + k)
                if abs(pred_id) > (n_left_prev or 0) + (n_right_prev or n_right):
                    pred_id = None
            if n_left_next is not None:
                succ_id = -(n_left_next + k)
                if abs(succ_id) > (n_left_next or 0) + (n_right_next or n_right):
                    succ_id = None
            w = chain_widths.get(k, LANE_WIDTH)
            if k == n_right:
                if r_pocket:
                    w = right_stable_w or chain_widths.get('pocket_right', w)
                    opening = r_pocket and not r_pocket_prev
                    closing = r_pocket and not r_pocket_next
                    if opening and not closing:
                        poly = _taper_poly(w, right_L_open, True)
                        pred_id = None
                    elif closing and not opening:
                        poly = _taper_poly(w, right_L_close, False)
                        succ_id = None
                    elif opening and closing:
                        poly = _taper_poly(w, right_L_open, True)
                        pred_id = None
                        succ_id = None
                        lane_sec_len = sec_len
                else:
                    opening = n_right_prev is not None and n_right > n_right_prev
                    closing = n_right_next is not None and n_right > n_right_next
                    if opening:
                        poly = _taper_poly(w, L_taper, True)
                        pred_id = None
                        succ_id = None if closing else succ_id
                        lane_sec_len = sec_len if closing else None
                    elif closing:
                        poly = _taper_poly(w, L_taper, False)
                        succ_id = None
            _add_lane(right, -i, pred_id, succ_id, width=w, poly=poly,
                      L_taper=L_taper if poly else None, sec_len=lane_sec_len)

def build_lanes(road_el, chain, a2_by_id, rev_91_rid, rev_91_node, by_l_linkid, by_from_node,
                b2_r, b2_l, b2_map, bilateral, nodes_7,
                opposite_chain=None):
    lanes_el     = ET.SubElement(road_el, "lanes")
    chain_widths = compute_chain_widths(chain, a2_by_id, rev_91_rid, b2_r, b2_l)

    n_opposite = opposite_widths = None
    if bilateral and opposite_chain:
        opposite_widths = compute_chain_widths(opposite_chain, a2_by_id, rev_91_rid, b2_r, b2_l)
        n_opposite = max((_link_lane_info(lk, a2_by_id, rev_91_rid, rev_91_node)[0]
                          for lk in opposite_chain), default=1)

    node_to_s, total_s = _build_node_to_s(chain, b2_map)
    spans = _find_pocket_spans(chain, a2_by_id, by_from_node, by_l_linkid, b2_r, b2_l, node_to_s)
    left_spans  = [p for p in spans if p['side'] == 'left']
    right_spans = [p for p in spans if p['side'] == 'right']
    if left_spans:
        chain_widths['pocket'] = round(sum(p['stable_w'] for p in left_spans)/len(left_spans), 2)
    if right_spans:
        chain_widths['pocket_right'] = round(sum(p['stable_w'] for p in right_spans)/len(right_spans), 2)

    pocket_s_bounds = {p['s_start'] for p in spans} | {p['s_end'] for p in spans}

    def _active_right(s):
        return next((p for p in right_spans if p['s_start'] <= s < p['s_end']), None)

    s, sections, prev_key = 0.0, [], None
    for lk in chain:
        n_right, n_left = _link_lane_info(lk, a2_by_id, rev_91_rid, rev_91_node)
        rp       = _active_right(s)
        at_node7 = str(lk.get('fromnodeid', '')) in nodes_7
        at_bound = s in pocket_s_bounds
        key = (n_right, n_left, id(rp) if rp else None)
        if key != prev_key or at_node7 or at_bound:
            sections.append((s, n_right, n_left, rp))
            prev_key = key
        pts = b2_map.get(str(lk['id']))
        if pts:
            s += _coords_length(pts)

    if not sections:
        return

    pocket_w = chain_widths.get('pocket', LANE_WIDTH)

    if not bilateral:
        prev_offset = 0.0
        for i, (sec_s, _, n_left, _rp) in enumerate(sections):
            offset = n_left * pocket_w
            next_sec_s = sections[i+1][0] if i+1 < len(sections) else total_s
            sec_len_lo = max(next_sec_s - sec_s, 1.0)
            if offset != prev_offset:
                lo_L = next((p['taper_open'] if abs(p['s_start']-sec_s) < 1.0
                             else p['taper_close']
                             for p in left_spans
                             if abs(p['s_start']-sec_s) < 1.0 or abs(p['s_end']-sec_s) < 1.0),
                            TAPER_LEN)
                L = min(sec_len_lo, lo_L)
                if i == 0:
                    ET.SubElement(lanes_el, "laneOffset",
                                  s=f"{sec_s:.6e}", a=f"{offset:.6e}", b="0", c="0", d="0")
                else:
                    a, b, c, d = _taper_poly(offset, L, True) if offset > prev_offset \
                                  else _taper_poly(prev_offset, L, False)
                    ET.SubElement(lanes_el, "laneOffset",
                                  s=f"{sec_s:.6e}", a=f"{a:.6e}", b=f"{b:.6e}", c=f"{c:.6e}", d=f"{d:.6e}")
                    if offset == 0 or L < sec_len_lo - 0.1:
                        ET.SubElement(lanes_el, "laneOffset",
                                      s=f"{sec_s+L:.6e}", a=f"{offset:.6e}", b="0", c="0", d="0")
                prev_offset = offset
    else:
        ET.SubElement(lanes_el, "laneOffset",
                      s="0.000000e+00", a="0.000000e+00", b="0", c="0", d="0")

    for i, (sec_s, n_right, n_left, rp) in enumerate(sections):
        n_left_prev  = sections[i-1][2] if i > 0 else None
        n_left_next  = sections[i+1][2] if i < len(sections)-1 else None
        n_right_prev = sections[i-1][1] if i > 0 else None
        n_right_next = sections[i+1][1] if i < len(sections)-1 else None
        rp_prev      = sections[i-1][3] if i > 0 else None
        rp_next      = sections[i+1][3] if i < len(sections)-1 else None
        next_sec_s   = sections[i+1][0] if i+1 < len(sections) else total_s
        sec_len      = max(next_sec_s - sec_s, 1.0)

        left_L_open = left_L_close = TAPER_LEN
        for p in left_spans:
            if p['s_start'] <= sec_s < p['s_end']:
                left_L_open, left_L_close = p['taper_open'], p['taper_close']
                break

        right_L_open = right_L_close = TAPER_LEN
        right_stable_w = chain_widths.get('pocket_right', LANE_WIDTH)
        if rp:
            right_L_open, right_L_close = rp['taper_open'], rp['taper_close']
            right_stable_w = rp['stable_w']

        _add_laneSection(lanes_el, sec_s, n_right, n_left, bilateral,
                         n_left_prev, n_left_next, chain_widths,
                         n_opposite, opposite_widths,
                         n_right_prev, n_right_next,
                         min(sec_len, left_L_open), sec_len,
                         rp is not None, rp_prev is not None, rp_next is not None,
                         right_stable_w, right_L_open, right_L_close)

def find_overlap(chain_coords):
    from shapely.geometry import LineString
    rids  = [rid for rid, _ in chain_coords]
    lines = [LineString(coords) for _, coords in chain_coords]
    bilateral, skip = {}, set()
    for i in range(len(rids)):
        if rids[i] in skip:
            continue
        for j in range(i+1, len(rids)):
            if rids[j] in skip:
                continue
            buf_i = lines[i].buffer(PARALLEL_DIST)
            buf_j = lines[j].buffer(PARALLEL_DIST)
            ratio_j = lines[j].intersection(buf_i).length / lines[j].length
            ratio_i = lines[i].intersection(buf_j).length / lines[i].length
            if min(ratio_i, ratio_j) > OVERLAP_THRESH:
                bilateral[rids[i]] = rids[j]
                skip.add(rids[j])
    return bilateral, skip

def convert():
    a1 = gpd.read_file(os.path.join(SHP_DIR, "A1_NODE.shp"))
    a2 = gpd.read_file(os.path.join(SHP_DIR, "A2_LINK.shp"))
    b2 = gpd.read_file(os.path.join(SHP_DIR, "B2_SURFACELINEMARK.shp"))
    a1.columns = [c.lower() for c in a1.columns]
    a2.columns = [c.lower() for c in a2.columns]
    b2.columns = [c.lower() for c in b2.columns]

    nodes_99   = set(a1[a1['nodetype'].astype(str) == '99']['id'].astype(str))
    nodes_7    = set(a1[a1['nodetype'].astype(str) == '7']['id'].astype(str))
    normal_to  = set(a2[a2['linktype'].astype(str) != '1']['tonodeid'].astype(str))
    stop_nodes = set(a2[a2['linktype'].astype(str) == '1']['fromnodeid'].astype(str)) & normal_to

    a2m = a2[~a2['linktype'].astype(str).isin(EXCLUDE_LINK)].copy()
    a2m = a2m[a2m['laneno'].astype(str) == '1'].copy()
    uturn_node = (a2m['fromnodeid'].astype(str).isin(nodes_99) &
                  a2m['tonodeid'].astype(str).isin(nodes_99))
    uturn_link = a2m['r_linkid'].isna() & a2m['l_linkid'].isna()
    a2m = a2m[~(uturn_node & uturn_link)].copy()

    a2_by_id, rev_91_rid, rev_91_node, by_l_linkid, by_from_node = build_a2_index(a2)
    b2_r, b2_l = build_b2_maps(b2, a2_by_id)

    a2m_ids = set(a2m['id'].astype(str))
    b2_map  = {k: v for k, v in b2_r.items() if k in a2m_ids}

    all_xy = [(c[0], c[1]) for g in a2m.geometry for c in g.coords]
    ox = min(x for x, _ in all_xy)
    oy = min(y for _, y in all_xy)

    all_to_nodes = set(a2m['tonodeid'].astype(str))
    adj = defaultdict(list)
    for _, r in a2m.iterrows():
        adj[str(r['fromnodeid'])].append(r)

    visited, road_id, node_to_road, chains = set(), 0, {}, []

    for _, start in a2m.iterrows():
        sid = str(start['id'])
        if sid in visited: continue
        fn = str(start['fromnodeid'])
        if fn not in stop_nodes and fn in all_to_nodes:
            continue
        chain = [start]
        visited.add(sid)
        cur_node = str(start['tonodeid'])
        while cur_node not in stop_nodes:
            nexts = [lk for lk in adj[cur_node] if str(lk['id']) not in visited]
            if len(nexts) != 1: break
            nxt = nexts[0]
            chain.append(nxt)
            visited.add(str(nxt['id']))
            cur_node = str(nxt['tonodeid'])
        road_id += 1
        node_to_road[fn] = road_id
        chains.append((road_id, chain, fn, cur_node))

    for _, row in a2m.iterrows():
        sid = str(row['id'])
        if sid in visited: continue
        road_id += 1
        fn = str(row['fromnodeid'])
        node_to_road[fn] = road_id
        chains.append((road_id, [row], fn, str(row['tonodeid'])))
        visited.add(sid)

    chain_data, chain_by_rid = [], {}
    for rid, chain, fn, tn in chains:
        coords = resolve_coords(chain, b2_r, a2m_ids)
        if coords:
            chain_data.append((rid, chain, fn, tn, coords))
            chain_by_rid[rid] = chain

    bilateral_map, skip_rids = find_overlap([(rid, coords) for rid, _, _, _, coords in chain_data])
    bilateral_rids = set(bilateral_map.keys())
    print(f"양방향 road: {len(bilateral_rids)}, skip: {len(skip_rids)}")

    od  = ET.Element("OpenDRIVE")
    hdr = ET.SubElement(od, "header", revMajor="1", revMinor="7",
                        name="NGII2XODR", version="1.0", date="2026-04-09")
    ET.SubElement(hdr, "geoReference").text = GEO_REF

    out_count = 0
    for rid, chain, fn, tn, coords in chain_data:
        if rid in skip_rids:
            continue

        road_el = ET.SubElement(od, "road", name=f"road{rid}", id=str(rid),
                                length="0", junction="-1", rule="RHT")
        lk_el = ET.SubElement(road_el, "link")
        pred, succ = node_to_road.get(fn), node_to_road.get(tn)
        if pred and pred != rid:
            ET.SubElement(lk_el, "predecessor", elementType="road",
                          elementId=str(pred), contactPoint="end")
        if succ and succ != rid:
            ET.SubElement(lk_el, "successor", elementType="road",
                          elementId=str(succ), contactPoint="start")

        L = build_planview(road_el, coords, ox, oy)
        road_el.set("length", f"{L:.6e}")

        opp_rid   = bilateral_map.get(rid)
        opp_chain = chain_by_rid.get(opp_rid) if opp_rid else None
        build_lanes(road_el, chain, a2_by_id, rev_91_rid, rev_91_node, by_l_linkid, by_from_node,
                    b2_r, b2_l, b2_map, rid in bilateral_rids, nodes_7,
                    opposite_chain=opp_chain)

        out_count += 1

    out_dir = os.path.dirname(OUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pretty = minidom.parseString(ET.tostring(od, 'utf-8')).toprettyxml(indent="  ")
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(pretty)
    print(f"완료: {OUT_PATH}  (roads={out_count})")

if __name__ == "__main__":
    convert()