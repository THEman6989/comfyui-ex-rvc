# ARCHITECTURE — comfyui-ex-rvc (Researcher Pipeline)

Komplexe Nodes und wie sie intern funktionieren.

---

## DINOv2FrameChangeDetector

**Datei:** `frame_embedding_change_detector.py`

### Zweck

Erkennt visuelle Outfit-Wechsel in Crop-Sequenzen mittels DINOv2-Embeddings.  
Ist NICHT für Audio-Beat-Erkennung (dafür BeatItNode).  
Erkennt, WO im Video sich das Outfit visuell ändert.

### Architektur

```
Crops (von MaskCropStabilizer)
        │
        ▼
┌───────────────────────────┐
│ DINOv2 Embedding          │
│ pro Frame: ViT → 768-dim  │  (vitb14)
│ L2-normalisiert           │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ Cosine Distance           │
│ dist[i] = 1 - dot(A[i], A[i+1]) │
│ Hoch = Frame-Paar unterschiedlich │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ Search Window             │
│ um beat_frame:            │
│ [beat-search_before,      │
│  beat+search_after]       │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ Peak Detection            │
│ dist > change_threshold?  │
│ → Lokales Maximum?        │
│ → Min-Gap eingehalten?    │
└───────────┬───────────────┘
            │
    ┌───────┴────────┐
    ▼                ▼
┌─────────┐   ┌──────────────────┐
│ Change  │   │ Kein Change      │
│ gefunden│   │ → Modus-abhängig │
└────┬────┘   └────────┬─────────┘
     │                 │
     ▼                 ▼
┌──────────────────────────────┐
│ Alignment                    │
│ final_drop = detected_change │
│   + alignment_offset         │
│   + manual_offset            │
└──────────────────────────────┘
```

### 3 Modi

| Modus | Change gefunden | Kein Change |
|-------|----------------|-------------|
| `detect_existing_change` | `has_existing=true` | `has_existing=false`, keine Generierung |
| `beat_only_generate_outfit_drop` | `has_existing=false` | `has_existing=false`, `needs_generated=true`, Drop = beat + offset |
| `auto` | `has_existing=true` | `needs_generated=true`, Drop = beat + offset |

### Alignment

Zwei Offsets, die addiert werden:

```
final_drop_frame = detected_change_frame
                 + alignment_offset_frames   (systematisch, z.B. +2 wenn Change immer 2 Frames nach Beat)
                 + manual_offset_frames      (manuell pro Video)
```

### Modell-Download

**Primary:** HuggingFace Hub via `transformers.AutoModel`  
→ `facebook/dinov2-small` (vits14, 384d)  
→ `facebook/dinov2-base` (vitb14, 768d) — **Default**  
→ `facebook/dinov2-large` (vitl14, 1024d)

Cache: `~/.cache/huggingface/hub/`  
Fallback: `torch.hub.load("facebookresearch/dinov2", ...)`

Download passiert automatisch beim ersten Use.  
Cached danach (~2s Ladezeit statt Download).

### Wichtige Inputs

| Input | Default | Bedeutung |
|-------|---------|-----------|
| `mode` | `auto` | detect_existing_change / beat_only / auto |
| `model_name` | `dinov2_vitb14` | vitb14 = beste Balance |
| `change_threshold` | 0.25 | Cosine-Distanz-Schwelle für Outfit-Wechsel |
| `beat_frame` | 0 | Beat-Frame für Search-Window-Zentrierung |
| `search_before_frames` | 6 | Frames VOR beat_frame durchsuchen |
| `search_after_frames` | 12 | Frames NACH beat_frame durchsuchen |
| `alignment_offset_frames` | 0 | Systematischer Offset |
| `manual_offset_frames` | 0 | Manueller Offset |

### Outputs (8)

```
change_json               → voller JSON mit allen Scores + Alignment
best_change_frame         → finaler Drop-Frame
last_old_outfit_frame     → letzter Frame vor Wechsel
first_new_outfit_frame    → erster Frame nach Wechsel
confidence                → 0.0-1.0
has_existing_visual_change → true/false
needs_generated_outfit_drop → true/false
report                    → Text-Log
```

---

## BeatChangeSynchronizer

**Datei:** `mask_researcher_tools.py`

### Zweck

Synchronisiert Audio-Beatdrops (FrameSequenceGenerator) mit visuellen Outfit-Wechseln (DINOv2Detector).  
Snapped jeden Beatdrop an den nächstgelegenen Change-Frame.

### Flow

```
beats_used (Audio-Beats)     change_frames (DINOv2-Changes)
        │                            │
        └──────────┬─────────────────┘
                   ▼
        ┌─────────────────────┐
        │ Pro Beatdrop:        │
        │  Suche nächsten      │
        │  Change-Frame        │
        │  innerhalb max_dist  │
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   ┌─────────┐          ┌──────────┐
   │ Gefunden│          │ Keiner   │
   │ → snap  │          │ → bleibt │
   └─────────┘          └──────────┘
                   │
                   ▼
        synced_beats (mit original_frame_index + sync_distance)
```

### 3 Snap-Modi

| Modus | Verhalten |
|-------|----------|
| `snap_nearest` | Nächster Change-Frame, egal ob vor oder nach Beat |
| `snap_before` | Nur Change-Frames VOR dem Beat |
| `snap_after` | Nur Change-Frames NACH dem Beat |

### Wichtige Inputs

| Input | Default | Bedeutung |
|-------|---------|-----------|
| `beats_used` | `[]` | JSON von FrameSequenceGenerator |
| `change_frames` | `[]` | JSON von DINOv2FrameChangeDetector |
| `max_distance` | 30 | Max Frames die ein Beat gesnapped werden darf |
| `mode` | `snap_nearest` | snap_nearest / snap_before / snap_after |

---

## FrameSequenceGenerator

**Datei:** `beatdrop_nodes.py`

### Zweck

Extrahiert Frames um Drop-Zeitpunkte aus einem IMAGE-Batch oder Video.  
Erzeugt `beats_used` JSON mit `batch_offset` + `batch_frame_count` —  
das ist der **Vertrag** zwischen FrameSequenceGenerator und allen downstream Nodes.

### beats_used Vertrag

Jeder Eintrag:
```json
{
  "beat_index": 0,
  "time_seconds": 1.0,
  "frame_index": 10,
  "is_drop": true,
  "energy_jump": 3.0,
  "range_start": 0.0,
  "range_end": 2.0,
  "batch_offset": 0,        // ← Start-Index im IMAGE-Batch
  "batch_frame_count": 21    // ← Anzahl Frames in diesem Fenster
}
```

`batch_offset` + `batch_frame_count` definieren exakt, welche Frames zu welchem  
Drop-Fenster gehören. BeatDropSelector, MaskInterpolatorPro, MaskCropStabilizer  
lesen diese Felder um Fenstergrenzen zu respektieren.

### Filter-Modi

| Input | Effekt |
|-------|--------|
| `drops_only=true` | Nur echte Drops (is_drop=true), keine normalen Beats |
| `main_job_only=true` | Nur der stärkste Drop (höchster energy_jump) |
| Beide false | Alle Beats/Drops |

---

## MaskInterpolatorPro + MaskCropStabilizer

**Datei:** `mask_researcher_tools.py`

### beats_used Integration

Beide Nodes respektieren Fenstergrenzen aus `beats_used`:

- **MaskInterpolatorPro**: Interpoliert leere Masken NUR innerhalb eines Drop-Fensters.  
  Keine Interpolation ÜBER Fenstergrenzen hinweg (würde alte+neue Maske vermischen).

- **MaskCropStabilizer**: Setzt EMA-Smoothing an Fenstergrenzen zurück.  
  Kein Smoothing von Crop-Positionen über den Outfit-Wechsel hinweg.

---

## Komplette Pipeline

```
BeatItNode (Audio-Analyse)
    │
    ▼
FrameSequenceGenerator (Frame-Extraktion + beats_used)
    │
    ▼
SAM3/SAM3.1 (Maskierung)
    │
    ▼
MaskQualityFilter (schlechte Masken erkennen → leeren)
    │
    ▼
MaskInterpolatorPro (leere Masken interpolieren, Fenster-respektierend)
    │
    ▼
MaskCropStabilizer (stabile Crops, Smoothing-Reset an Fenstergrenzen)
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
DINOv2FrameChangeDetector          (direkt zu BeatDropSelector
(visueller Outfit-Wechsel)          wenn kein Driving-Video)
    │
    ▼
BeatChangeSynchronizer
(snapped Audio-Beats an visuelle Changes)
    │
    ▼
BeatDropSelectorNode (ComfyUI-ImageSelector-LLM)
(Frame-Selektion pro Fenster, Re-Ranker, History, Diversity)
    │
    ▼
AlphaRavisOutfitReferenceJudgeNode (ComfyUI-ImageSelector-LLM)
(Vision-LLM: Scene-Fit + Change-Strength + Beatdrop-Impact)
    │
    ▼
ResearcherPlanWriter (TODO)
```
