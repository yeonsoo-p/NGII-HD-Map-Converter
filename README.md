# NGII SHP → OpenDRIVE Converter

국토지리정보원(NGII) 정밀도로지도 SHP 파일을 CarMaker 호환 OpenDRIVE(`.xodr`) 포맷으로 변환하는 Python 스크립트.

---

## 입력 데이터

| 레이어 | 파일명 | 사용 목적 |
|---|---|---|
| A1 | `A1_NODE.shp` | 교차로/유턴 노드 타입 |
| A2 | `A2_LINK.shp` | 도로 링크 (차로 구조, 연결관계) |
| B2 | `B2_SURFACELINEMARK.shp` | 차선 표시 → 차로폭 측정 |

- 좌표계: **EPSG:5186** (한국 TM, NGII 표준)
- CarMaker origin offset: 전체 좌표 최솟값을 원점으로 자동 설정 (`<header><offset>` 기록)

---

## 출력

```
C:\CM_Projects\3. AV_VIL\Data\Road\output_A1A2B2.xodr
```

OpenDRIVE 1.7 포맷. CarMaker IPG Road Editor에서 직접 불러올 수 있음.

---

## 주요 처리 흐름

```
SHP 로드
  └─ A2 필터링 (linktype=1 교차로링크 제외, laneno=1 메인차로만)
       └─ 유턴 링크 제거 (nodetype=99 + R/L_LinkID 없음)
            └─ 링크 체이닝 (stop_node 기준으로 교차로 구간 분할)
                 └─ B2 planView 좌표 매핑
                      └─ 양방향 도로 검출 (buffer overlap ≥ 0.8)
                           └─ 각 road별 planView + lanes 생성
                                └─ .xodr 저장
```

---

## 핵심 기능

### planView 기하 처리
- **직선 판별**: 중간점의 평균 수직거리 ≤ `STRAIGHT_THR(0.5m)` → `<line>`
- **곡선 단순화**: Ramer-Douglas-Peucker (`ε=0.3m`) 후 Catmull-Rom 스플라인 → `<paramPoly3>`

### 차로폭 산출 (B2 기반)
- B2 우측 차선(R_LinkID) 좌표를 `LineString`으로 구성
- 좌측 차선 각 정점에서 우측까지 수직거리 평균 → 차로폭
- 유효 범위: `WIDTH_MIN(1.0m) ~ WIDTH_MAX(5.0m)`
- B2 미검출 시 기본값 `LANE_WIDTH=3.5m` 사용

### 다차로 구조
- `R_LinkID` 체이닝으로 우측 차로 수(`n_right`) 결정
- `L_LinkID` / `LaneNo≥91`로 좌측 포켓차로(`n_left`) 결정
- 각 차로별 독립 폭 적용 (`chain_widths[k]`)

### 포켓차로 (LaneNo ≥ 91)
- 좌측: `from_node`에서 시작하는 LaneNo≥91 링크 탐지
- 우측: `by_l_linkid`로 역방향 탐색
- `s_start / s_end` 구간에 taper 폭 다항식 (`_taper_poly`) 적용
- taper 길이는 B2 정점폭 분포에서 자동 추정 (fallback: `TAPER_LEN=20.0m`)

### 양방향 도로 (bilateral)
- 두 road의 geometry를 `buffer(1.0m)` 후 교차 비율 ≥ `OVERLAP_THRESH(0.8)` → 양방향 판정
- 반대 방향 chain의 차로를 `<left>` 섹션에 배치

### laneOffset
- 포켓차로 존재 시 `laneOffset` 스텝 상수로 기록 (taper 미적용, 향후 개선 예정)
- 일반 구간: `a=0, b=0, c=0, d=0`

### pred/succ 연결
- `node_to_road` 딕셔너리로 FromNode/ToNode → road ID 매핑
- `<predecessor>` / `<successor>` contactPoint 자동 설정

---

## 설정 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `SHP_DIR` | `Daegu` 경로 | 입력 SHP 디렉터리 |
| `OUT_PATH` | `output_A1A2B2.xodr` | 출력 경로 |
| `LANE_WIDTH` | `3.5` m | B2 미검출 시 기본 차로폭 |
| `STRAIGHT_THR` | `0.5` m | 직선 판별 평균 편차 임계값 |
| `RDP_EPSILON` | `0.3` m | RDP 단순화 허용 오차 |
| `PARALLEL_DIST` | `1.0` m | 양방향 판별 buffer 반경 |
| `TAPER_LEN` | `20.0` m | taper 기본 길이 (B2 추정 실패 시) |
| `OVERLAP_THRESH` | `0.8` | 양방향 판별 overlap 비율 |
| `WIDTH_MIN/MAX` | `1.0 / 5.0` m | 유효 차로폭 범위 |

---

## 의존성

```bash
pip install geopandas shapely
```

Python 표준 라이브러리: `math`, `os`, `xml.etree.ElementTree`, `xml.dom.minidom`, `collections`

---

## 실행

```bash
python shp2xodr_converter.py
```

SHP_DIR 내 `A1_NODE.shp`, `A2_LINK.shp`, `B2_SURFACELINEMARK.shp` 세 파일이 존재해야 함.

---

## 현재 한계 / 향후 개선 사항

- `laneOffset` taper 미적용 (현재 step-constant)
- `TAPER_LEN` 고정값 사용 → B2 실측 taper 길이로 대체 필요
- 포켓차로 폭을 A1 노드 데이터에서 추가 보완
- merge lane ↔ pocket lane 간 pred/succ 연결 미구현
- 지원 지역: 대구(Daegu) / 판교 제로시티 / 상암 (SHP_DIR 주석 변경)

---

## 파일 구조 참고

```
shp2xodr_converter.py
├── convert()                  # 메인 진입점
├── build_planview()           # <planView> 기하 생성
├── build_lanes()              # <lanes> 전체 구성
│   ├── _find_pocket_spans()   # 포켓차로 구간 탐지
│   ├── compute_chain_widths() # B2 기반 차로폭 계산
│   ├── _add_laneSection()     # <laneSection> XML 생성
│   └── _add_lane()            # <lane> XML 생성
├── build_b2_maps()            # B2 좌표 체이닝
├── find_overlap()             # 양방향 도로 검출
└── build_a2_index()           # A2 역방향 인덱스 구성
```
