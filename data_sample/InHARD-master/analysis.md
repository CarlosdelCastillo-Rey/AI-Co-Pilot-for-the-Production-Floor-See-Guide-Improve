# InHARD — dataset analysis context

This document consolidates **README**, **on-disk layout**, **annotation schema**, and **statistics from this workspace copy** so you can design models, loaders, and evaluation without re-reading scattered sources.

**Primary references**

- Official README: `README.md` in this folder  
- Paper: [IEEE Xplore 9209531](https://ieeexplore.ieee.org/document/9209531)  
- Full archive (if you need complete assets): [Zenodo record 4003541](https://zenodo.org/record/4003541)  
- Laboratory page: [LINEACT-CESI — InHARD](https://recherche.cesi.fr/inhard-industrial-human-action-recognition-dataset/)  
- License: `LICENSE` (MIT, Copyright 2019 vhavard)

---

## 1. What InHARD is

**InHARD** (Industrial Human Action Recognition Dataset) is a **multimodal industrial HAR** corpus: synchronized **RGB video** and **mocap-style skeleton** data from a real assembly-style workflow, aimed at human action analysis in **human–robot collaboration** settings.

**Scale (paper / README claims)**

- Over **2M frames** total, **16 subjects** in the full release narrative (this copy uses **38 session recordings** — subject + recurrence, e.g. `P01_R01`; see §6).  
- **13 meta-action** classes and **74 fine-grained action** classes (canonical list in `rsc/Action-Meta-action-list.xlsx`).  
- **4800+** action samples in the full dataset; **this copy’s segmented index** aligns **5303** clip rows with **5303** paired `.mp4` / `.bvh` files (§5).

**Tasks you might build**

- **Meta-action recognition** (13 + background): pick/place, assemble, consult sheets, no-action, etc.  
- **Fine-grained action recognition** (`Action_label` strings, many variants per operation).  
- **Temporal segmentation** / online detection on full sessions (`Online/`).  
- **Multimodal fusion** (RGB + skeleton), **cross-view** reasoning (three camera tiles in one frame).

---

## 2. Modalities and capture

### 2.1 Skeleton (BVH)

- **Sensor**: Perception Neuron 32 v2 (as stated in README).  
- **Nominal rate**: **120 Hz** (README); **measured from BVH** in this copy: `Frame Time ≈ 0.008401 s` → **≈119 Hz** (`1 / 0.008401…`). Treat as **~120 Hz** for modeling; use file metadata if you need exact timing.  
- **Representation**: BVH hierarchy with **6 channels per joint** (root: `Xposition Yposition Zposition Yrotation Xrotation Zrotation`; same pattern on children). Joints follow a standard body tree (Hips → legs, spine chain, arms, etc.); README points to `rsc/Skeleton-joints-hierarchy.png`.  
- **Tooling**: [PyMO](https://github.com/omimo/PyMO/) is recommended in the README for BVH I/O.

### 2.2 RGB video

- **Capture**: three **Logitech C920** streams composited into **one MP4 per session or clip**: top, left (−45°), right (+45°) — layout described in README (`rsc/InHard_dataset.png`).  
- **Effective RGB frame rate** (derived from `Action_*_rgb_frame` and `Duration_sec` in `InHARD.csv`): **≈30 fps** (spot checks ~29.7–30.1 fps).  
- **Important**: One “RGB frame” is a **three-view mosaic**; any vision model should account for **layout / ROI** or crop per view if you simulate separate cameras.

---

## 3. Repository layout (`InHARD-master/`)

```
InHARD-master/
├── README.md
├── LICENSE
├── analysis.md              ← this file
├── rsc/
│   ├── Action-Meta-action-list.xlsx   # meta-action ↔ fine action taxonomy
│   ├── InHard_dataset.png
│   └── Skeleton-joints-hierarchy.png
└── 01-InHARD/
    ├── Online/              # full-take alignment
    │   ├── InHARD.csv
    │   ├── InHARD_All_No_Action.csv   # includes explicit "No action" segments
    │   ├── RGB/             # P##_R##.mp4 (38 files)
    │   ├── Skeleton/      # P##_R##.bvh (38 files)
    │   └── Labels/        # P##_R##.anvil + Annotation_specs/
    └── Segmented/         # one clip per CSV row (same row order as index)
        ├── InHARD.csv     # 5303 rows + header; includes No action
        ├── RGBSegmented/<Meta_action_label>/P##_R##_<t0>_<t1>.mp4
        ├── SkletonSegmented/<Meta_action_label>/P##_R##_<t0>_<t1>.bvh  # note typo "Skleton"
        └── Labels/        # same ANVIL structure as Online
```

**Naming convention (segmented clips)**

- Pattern: `{File}_{Action_start_rgb_sec}_{Action_end_rgb_sec}.mp4` / `.bvh`  
- Example: `P01_R01_0013.84_0018.88` — session `P01_R01`, clip from **13.84 s** to **18.88 s** in RGB timeline.  
- Clips are filed under folders named by **`Meta_action_label`** (human-readable class).

---

## 4. Annotation levels

### 4.1 Meta-actions (coarse, 13 + background)

Used for **high-level HAR** and folder organization. In **`InHARD.csv`** the column is `Meta_action_label`; **`Meta_action_class_number`** is an integer ID (see §7 for ID quirks).

**Meta-action labels present in this copy’s `Segmented/InHARD.csv`**

| `Meta_action_class_number` | `Meta_action_label`   |
|---------------------------:|------------------------|
| 0 | No action |
| 2 | Consult sheets |
| 3 | Turn sheets |
| 4 | Take screwdriver |
| 5 | Put down screwdriver |
| 6 | Picking in front |
| 7 | Picking left |
| 8 | Take measuring rod |
| 9 | Put down measuring rod |
| 10 | Take component |
| 11 | Put down component |
| 12 | Assemble system |
| 13 | Take subsystem |
| 14 | Put down subsystem |

**Class `1` does not appear** in the CSVs here; **`0` is explicit background** (“No action”). When merging with the spreadsheet taxonomy, see §7.

**Rough class frequencies** (`Segmented/InHARD.csv`, row counts per meta-label)

- Assemble system: 1378  
- Picking left: 641  
- No action: 500  
- Take component: 485  
- Picking in front: 456  
- Take screwdriver: 420  
- Put down screwdriver: 416  
- Put down component: 385  
- Turn sheets: 224  
- Consult sheets: 132  
- Put down subsystem: 77  
- Take measuring rod: 76  
- Put down measuring rod: 74  
- Take subsystem: 39  

Strong **imbalance** toward “Assemble system” and picking-like classes; plan **sampling, weighting, or metrics** (macro-F1, balanced acc) accordingly.

### 4.2 Fine-grained actions (`Action_label`)

- **String labels** prefixed by operation, e.g. `[OP010] Consult sheets`, `[OP030] Catch Fixture key LARD (x2)`.  
- **~103 distinct `Action_label` values** in this copy’s segmented CSV (exact count can drift if labels are edited).  
- Full controlled vocabulary and grouping live in **`rsc/Action-Meta-action-list.xlsx`** (sheet maps **Meta action label** → many **Action label** rows) and in **`01-InHARD/Segmented/Labels/Annotation_specs/Annotation_specs.xml`** (ANVIL value sets per operation).

### 4.3 Operations (`Operation`)

Sessions are blocks like **`OP010` … `OP070`** plus **`OP000`** appearing in this CSV export. These are **different assembly procedures**; the same meta-action (e.g. “Picking left”) recurs across operations with **different fine labels**.

---

## 5. CSV files — which to use

All tables share the same **logical timing**: for each row, BVH frame indices and RGB frame indices / seconds delimit one segment.

### 5.1 Columns (this copy)

**`01-InHARD/Online/InHARD.csv`** (4803 rows)

- No leading unnamed index column.  
- **Excludes** `Meta_action_class_number == 0` (no “No action” rows).

Columns:

`Action_end_bvh_frame`, `Action_end_rgb_frame`, `Action_end_rgb_sec`, `Action_label`, `Action_start_bvh_frame`, `Action_start_rgb_frame`, `Action_start_rgb_sec`, `Duration_sec`, `File`, `Meta_action_class_number`, `Meta_action_label`, `Operation`, `Recurrence`, `Subject`

**`01-InHARD/Segmented/InHARD.csv`** and **`01-InHARD/Online/InHARD_All_No_Action.csv`** (5303 / 7732 rows)

- Leading empty column name `""` from export (pandas-style index) — **drop or rename** when parsing.  
- **`Segmented`**: includes **No action** (500 rows with class 0 in this copy).  
- **`All_No_Action`**: many more rows; **2929** are “No action” — use when training **background models** or **dense temporal** labeling on full timelines.

### 5.2 Alignment rules for models

- **`File`**: session key (`P##_R##`), matches `Online/RGB/P##_R##.mp4` and `Online/Skeleton/P##_R##.bvh`.  
- **Segmented clips**: same `File` + time window in filename; one row ↔ one file pair under `RGBSegmented/…` and `SkletonSegmented/…`.  
- **BVH vs RGB**: use **`Action_*_bvh_frame`** for skeleton sequence indexing and **`Action_*_rgb_frame`** / `*_rgb_sec` for video. Ratios differ (~120 vs ~30); **resample or interpolate** explicitly for late fusion.

---

## 6. Sessions (files) in this copy

- **38 unique `File` values** in the CSVs (e.g. `P01_R01` … `P16_R02`).  
- **16 `Subject` IDs** (`P01`–`P16`), **38 sessions** when counting recurrences `R01`, `R02`, …  
- **8 `Operation` codes** observed: `OP000`, `OP010`, `OP020`, `OP030`, `OP040`, `OP050`, `OP060`, `OP070`.

---

## 7. Spreadsheet IDs vs `Meta_action_class_number`

`rsc/Action-Meta-action-list.xlsx` uses an **`ID` column** starting at **0 = No action**, **1 = Consult sheets**, **2 = Turn sheets**, … through **13 = Put down subsystem**.

The **`Meta_action_class_number` in the CSVs** matches **meta-action semantics** but uses **CSV IDs `0, 2, 3, …, 14`**, i.e. **CSV = spreadsheet `ID` + 1** for all non–No-action meta-actions, and **`0` stays No action**. Equivalently: spreadsheet `ID` 0 ↔ CSV 0; spreadsheet `ID` k ≥ 1 ↔ CSV **k + 1**. The integer **`1` is unused** in the CSVs inspected here — **reserve or remap** if you join to the XLSX by raw ID.

---

## 8. Official train / validation split and expertise

From **`README.md`** (reproduced for convenience):

**Experts vs beginners**

- **Experts**: subjects whose **average total action duration** for the whole manipulation is **under 6 minutes**; others are **beginners**.  
- README lists **bold** sessions as experts inside `S_train` and `S_val`.

**Training set `S_train`**

`P01_R01`, `P01_R03`, `P03_R01`, `P03_R03`, `P03_R04`, `P04_R02`, `P05_R03`, `P05_R04`, `P06_R01`, `P07_R01`, `P07_R02`, `P08_R02`, `P08_R04`, `P09_R01`, `P09_R03`, `P10_R01`, `P10_R02`, `P10_R03`, `P11_R02`, `P12_R01`, `P12_R02`, `P13_R02`, `P14_R01`, `P15_R01`, `P15_R02`, `P16_R02`

**Validation set `S_val`**

`P01_R02`, `P02_R01`, `P02_R02`, `P04_R01`, `P05_R01`, `P05_R02`, `P08_R01`, `P08_R03`, `P09_R02`, `P11_R01`, `P14_R02`, `P16_R01`

**Experts (bold in README)**  
`P03_R03`, `P07_R01`, `P07_R02`, `P08_R04`, `P09_R01`, `P10_R03`, `P14_R01`, `P15_R01`, `P15_R02`, `P02_R02`, `P08_R03`, `P09_R02`, `P14_R02`

For **subject-independent** evaluation, **hold out entire `File` sessions**, not random clips, to avoid leakage.

---

## 9. Labels (ANVIL)

- Original annotation / editing: **[ANVIL](https://www.anvil-software.org/)**.  
- **`Labels/*.anvil`** per session; **`Annotation_specs.xml`** defines allowed `Action_label` strings and colors per operation.  
- Use ANVIL if you need to **revise spans** or export alternate schemes.

---

## 10. Modeling notes (practical)

1. **Two temporal granularities**: **online** full takes vs **offline** pre-cut clips — choose one pipeline or both for pretrain/finetune.  
2. **Background class**: use **`InHARD_All_No_Action.csv`** or **`Segmented/InHARD.csv`** rows with `Meta_action_label == "No action"`; **`Online/InHARD.csv` omits** those segments.  
3. **Multimodal sync**: resample skeleton to RGB (e.g. strided 4:1) or use two encoders with **learned alignment**; document which frame is **inclusive** (`start`/`end` columns — verify off-by-one in your loader).  
4. **Hierarchy**: use **meta-action** for robust classes; use **`Action_label`** for fine-grained or **operation-conditioned** heads (`Operation` as side information or separate adapters).  
5. **Composite video**: consider **splitting tiles** into three pseudo-cameras for standard video backbones, or **ROI crops** around hands/tools if you add detection later.  
6. **Typos / consistency**: segmented skeleton folder is spelled **`SkletonSegmented`** on disk; README refers to **`Skeleton/`** in the Zenodo zip — code should use **actual path** on disk.

---

## 11. Citation (from README)

```bibtex
@inproceedings{dallel_inhard_2020,
	title = {{InHARD} - {Industrial} {Human} {Action} {Recognition} {Dataset} in the {Context} of {Industrial} {Collaborative} {Robotics}},
	doi = {10.1109/ICHMS49158.2020.9209531},
	booktitle = {2020 {IEEE} {International} {Conference} on {Human}-{Machine} {Systems} ({ICHMS})},
	author = {Dallel, Mejdi and Havard, Vincent and Baudry, David and Savatier, Xavier},
	month = sep,
	year = {2020},
	keywords = {RNN, Skeleton, Industry 4.0, LSTM, HAR, learning (artificial intelligence), Task analysis, Cameras, actual action recognition datasets, actual industrial human actions, business productivity, Collaboration, data analysis, Dataset, Deep Learning, health related actions, Human Action Recognition, human actions analysis, human robot collaboration, human robot collaborations, Human-Robot Collaboration (HRC), human-robot interaction, industrial action classes, Industrial collaborative robotics, industrial environment, Industrial Human Action Recognition, Industrial Human Action Recognition Dataset, machine learning algorithms, mutual actions, pose estimation, product quality, RGB+D, robot vision, Sensors, Service robots, dataset},
	pages = {1--6},
}
```

---

## 12. README content merged (source of truth for prose claims)

The following points are taken verbatim in meaning from `README.md`:

- Dataset name and purpose: industrial HAR with RGB + skeleton, real setting, HRC context.  
- Skeleton: 17 joints with **Tx, Ty, Tz** and **Rx, Ry, Rz** (BVH encodes these via joint channels / hierarchy).  
- RGB: three cameras, composite layout (camera 1 top-left, 2 top-right, 3 bottom-right).  
- **Action-Meta-action-list.xlsx** lists **13 meta-actions** and **74 actions**.  
- **`InHARD.csv`** is the main index for filenames, subject, operation, low/high labels, BVH/RGB start/end, duration.  
- Suggested protocol: **expert/beginner** split by **6-minute** average manipulation time rule; **`S_train` / `S_val`** lists above.  

If anything in this `analysis.md` disagrees with the **Zenodo archive** or the **paper**, prefer those for publication; this file reflects **this git snapshot** and computed checks where noted.
