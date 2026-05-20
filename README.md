# NGII SHP → OpenDRIVE Converter

국토지리정보원(NGII) 정밀도로지도 SHP 파일을 CarMaker 호환 **OpenDRIVE 1.7(.xodr)** 형식으로 변환하는 Python 스크립트입니다.

---

## 배경 지식

### OpenDRIVE란?

자율주행 시뮬레이터(CarMaker, CARLA 등)에서 사용하는 도로 기술 표준 XML 포맷입니다. 도로의 기하(planView), 차선 구조(lanes), 도로 간 연결관계(link)를 수치로 기술합니다.

### NGII 정밀도로지도란?

국토지리정보원이 제공하는 정밀도로지도로, 여러 레이어(SHP 파일)로 구성됩니다. 이 스크립트는 그 중 3개 레이어를 사용합니다.

| 레이어 | 파일                     | 담고 있는 정보                                     |
| ------ | ------------------------ | -------------------------------------------------- |
| A1     | `A1_NODE.shp`            | 교차로·유턴 등 노드 타입 정보                      |
| A2     | `A2_LINK.shp`            | 도로 링크 — 차로 번호, 연결 관계, 방향             |
| B2     | `B2_SURFACELINEMARK.shp` | 노면 차선 마킹 — 차로폭 측정 및 planView 좌표 원천 |

### A2 링크 구조 이해

A2는 차로 단위로 링크가 나뉩니다.

- `LaneNo=1` : 가장 왼쪽(기준) 차로
- `LaneNo=2, 3, …` : `R_LinkID`로 오른쪽 차로를 체이닝
- `LaneNo≥91` : 포켓차로(버스정류장·우회전 전용 등 비정규 차로)
- `LinkType=1` : 교차로 내부 경로 (본 스크립트에서 제외)

---

## 폴더 구조

```
shp2xodr_converter.py
shp_road/
├── Daegu/
│   ├── A1_NODE.shp  (+ .dbf, .prj, .shx)
│   ├── A2_LINK.shp
│   └── B2_SURFACELINEMARK.shp
├── Pangyo_Zerocity/
└── SangAm/
output_A1A2B2.xodr   ← 실행 후 생성됨
```

---

## 설치 및 실행

```bash
pip install geopandas shapely
python shp2xodr_converter.py
```

지역 변경은 스크립트 상단 `SHP_DIR` 주석을 바꿔서 선택합니다.

```python
SHP_DIR  = os.path.join(BASE, "shp_road", "Daegu")
# SHP_DIR  = os.path.join(BASE, "shp_road", "Pangyo_Zerocity")
# SHP_DIR  = os.path.join(BASE, "shp_road", "SangAm")
```

---

## 처리 흐름

```
① SHP 로드 (A1, A2, B2)
      ↓
② A2 필터링
   - LinkType=1 (교차로 내부) 제외
   - LaneNo=1 (기준 차로)만 선택
   - 유턴 링크 제거
      ↓
③ stop_node 추출 (교차로 경계 노드)
      ↓
④ 링크 체이닝 → road 단위 구성
      ↓
⑤ B2로 planView 좌표 생성
      ↓
⑥ 양방향 도로 검출
      ↓
⑦ 각 road마다 planView + lanes XML 생성
      ↓
⑧ output_A1A2B2.xodr 저장
```

---

## 핵심 처리 상세

### ② A2 필터링 및 유턴 제거

교차로 내부 경로(`LinkType=1`)와 기준 차로 외(`LaneNo≠1`)를 제거합니다.
유턴 링크 조건: FromNode·ToNode 모두 `NodeType=99`이면서 `R_LinkID`, `L_LinkID`가 모두 없는 링크.

### ③ stop_node — 교차로 경계 판별

`LinkType=1` 링크의 `FromNodeID` 중에서, 일반 링크의 `ToNodeID`와 겹치는 노드를 stop_node로 지정합니다. 체이닝이 이 노드에 도달하면 하나의 road가 끝납니다.

### ④ 링크 체이닝

stop_node에서 시작하여 `FromNode → ToNode` 방향으로 링크를 이어 붙입니다.

| 체이닝 종료 조건     | 이유                    |
| -------------------- | ----------------------- |
| stop_node 도달       | 교차로 직전 경계        |
| 다음 링크가 2개 이상 | 분기점 — 방향 결정 불가 |
| 다음 링크가 0개      | 물리적 끝점             |

### ⑤ planView 생성 (도로 기하)

B2의 `R_LinkID`를 기준으로 A2 링크에 차선 마킹 좌표를 매핑합니다.
B2 세그먼트가 여러 개인 경우, A2 geometry에 각 세그먼트의 중점을 투영해 공간 순서를 정렬하고 방향을 맞춥니다.

정렬된 좌표로 기하를 생성합니다.

- **직선 판별**: 중간 정점들의 평균 수직거리 ≤ `STRAIGHT_THR(0.5m)` → `<line>`
- **곡선**: Ramer-Douglas-Peucker(`ε=0.3m`)로 단순화 후 Catmull-Rom 스플라인 → `<paramPoly3>`

좌표 원점은 전체 A2 좌표의 최솟값(ox, oy)으로 자동 설정됩니다. CarMaker가 큰 절대좌표를 처리하지 못하기 때문입니다.

### ⑥ 양방향 도로 검출

왕복 2차선 도로는 A2에 방향별로 별도의 링크가 있습니다. 두 road의 geometry를 `1.0m` buffer로 팽창시켜, 양쪽 교차 비율이 모두 `0.8` 이상이면 같은 도로의 반대 방향으로 판정합니다. 한 쪽을 `<left>` + `<right>`로 통합하고, 나머지는 출력에서 제외합니다.

### ⑦ 차선(lanes) 생성

**차로폭 계산**

B2 좌측 차선(left_coords) 각 정점에서 우측 차선(right_coords) LineString까지의 수직거리를 평균 내어 차로폭으로 사용합니다. 유효 범위(`1.0m ~ 5.0m`) 외의 값은 제외하며, B2 데이터가 없으면 기본값 `3.5m`를 사용합니다.

**차로 수 결정**

- 우측 차로 수(`n_right`): `R_LinkID`를 끝까지 체이닝해 마지막 `LaneNo` 값으로 결정
- 좌측 포켓차로 수(`n_left`): `LaneNo≥91` 링크를 역방향 탐색해 결정

**laneSection 분할**

차로 수 변화 또는 `NodeType=7` 노드(합류점 등) 도달 시 새 laneSection을 추가합니다.

**차선 ID 배치**

단방향 도로 기준:

| 차선 종류                 | lane id                             |
| ------------------------- | ----------------------------------- |
| 좌측 포켓차로 (LaneNo≥91) | `-1` ~ `-n_left`                    |
| 일반 차로 (LaneNo=1~N)    | `-(n_left+1)` ~ `-(n_left+n_right)` |

양방향 도로는 반대 방향 차로를 `<left>` 섹션(`+1, +2, …`)에 배치합니다.

**포켓차로 taper**

포켓차로 진입/진출부에 Hermite 3차 다항식으로 자연스러운 폭 변화를 표현합니다.
taper 길이는 B2 정점 폭 분포에서 자동 추정하며, 추정 실패 시 `TAPER_LEN=20.0m`를 사용합니다.

**laneOffset**

좌측 포켓차로가 있는 구간에서는 center line을 우측으로 이동시키기 위해 `laneOffset`을 Hermite taper로 기록합니다.

**predecessor / successor 연결**

`node_to_road` 딕셔너리로 FromNode/ToNode → road ID를 매핑해 `<predecessor>`, `<successor>`를 자동으로 연결합니다.

---

## 파라미터 설정

| 파라미터         | 기본값 | 설명                            |
| ---------------- | ------ | ------------------------------- |
| `LANE_WIDTH`     | 3.5 m  | B2 실측 불가 시 fallback 차로폭 |
| `STRAIGHT_THR`   | 0.5 m  | 직선 판별 평균 수직거리 임계값  |
| `RDP_EPSILON`    | 0.3 m  | 곡선 단순화 허용 오차           |
| `PARALLEL_DIST`  | 1.0 m  | 양방향 판별 buffer 반경         |
| `OVERLAP_THRESH` | 0.8    | 양방향 판별 교차 비율 임계값    |
| `TAPER_LEN`      | 20.0 m | 포켓차로 taper 기본 길이        |
| `WIDTH_MIN`      | 1.0 m  | 유효 차로폭 하한                |
| `WIDTH_MAX`      | 5.0 m  | 유효 차로폭 상한                |

---

## 미구현 / 알려진 한계

- Junction element 미생성 (교차로 내부 연결 없음)
- `elevationProfile` 미지원 (고도 정보 무시)
- 포켓차로 폭을 A1 노드 데이터로 보완하는 기능 미구현
- merge lane ↔ pocket lane 간 predecessor/successor 연결 미구현
- 원형교차로 연결 도로 누락 가능성 있음
