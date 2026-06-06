# comfyui-ex-rvc

Custom nodes for ComfyUI.

## Load Image (API Push)

Node display name:

```text
Load Image (API Push)
```

Internal node key:

```text
ApiPushedLoadImage
```

This node works like a Load Image source, but the image is supplied through an API endpoint. By default it requires a fresh API push for each queue run, so an old image is not silently reused by the next run.

It also supports ComfyUI's normal Load Image upload style:

- endpoint: `POST /upload/image`
- multipart field: `image`
- optional form fields: `type`, `subfolder`, `overwrite`
- response fields: `name`, `subfolder`, `type`

### Upload An Image

If ComfyUI is running on `127.0.0.1:8188`, upload a raw image body:

```bash
curl -X POST \
  -H "Content-Type: image/png" \
  --data-binary @/path/to/image.png \
  http://127.0.0.1:8188/ex-rvc/api/pushed-image
```

Or upload as multipart form data:

```bash
curl -X POST \
  -F "image=@/path/to/image.png" \
  http://127.0.0.1:8188/ex-rvc/api/pushed-image
```

The custom endpoint accepts the same multipart field names as ComfyUI's official `/upload/image` endpoint:

```bash
curl -X POST \
  -F "image=@/path/to/image.png" \
  -F "type=input" \
  -F "subfolder=my_api_run" \
  -F "overwrite=false" \
  http://127.0.0.1:8188/ex-rvc/api/pushed-image
```

Example response:

```json
{
  "ok": true,
  "available": true,
  "pending": true,
  "image_id": "my_api_run/image_6b4f7a9d",
  "name": "image_6b4f7a9d.png",
  "subfolder": "ex_rvc_api_push/my_api_run",
  "type": "input",
  "filename": "image_6b4f7a9d.png",
  "size": 123456,
  "updated_at": "2026-05-22T12:00:00+00:00"
}
```

You can also use ComfyUI's official endpoint directly:

```bash
curl -X POST \
  -F "image=@/path/to/image.png" \
  -F "type=input" \
  -F "overwrite=true" \
  http://127.0.0.1:8188/upload/image
```

Then set the node's `source_mode` to `use_comfy_upload_image` and select the uploaded file in the `image` field.

### Node Modes

`source_mode = require_new_api_push`

This is the default. The node loads the pending API image once and then clears the pending marker. The image file is kept for reproducibility, but it will not be reused automatically in the next queue run.

If no fresh image was pushed before queueing, the node blocks with an error instead of loading an old image.

`source_mode = use_saved_image_id`

Use this only when you intentionally want to reload a specific uploaded image. Paste the returned `image_id` into the node's `image_id` field.

`source_mode = use_comfy_upload_image`

Use this when an external tool uploads through ComfyUI's official `/upload/image` endpoint, or when you want the node to behave like the normal Core `Load Image` node. The `image` field uses ComfyUI's standard `image_upload` widget.

### Check Pending Status

```bash
curl http://127.0.0.1:8188/ex-rvc/api/pushed-image
```

If a fresh upload is waiting for the next run, `pending` is `true`.

### Clear Pending Image

Clear only the pending marker:

```bash
curl -X DELETE http://127.0.0.1:8188/ex-rvc/api/pushed-image
```

Delete a saved image file by ID and also clear the pending marker:

```bash
curl -X DELETE "http://127.0.0.1:8188/ex-rvc/api/pushed-image?image_id=6b4f7a9df5e24b289d0edc2f0e8c2d91"
```

### Typical Flow

1. Add `Load Image (API Push)` to the workflow.
2. Keep `source_mode` set to `require_new_api_push`.
3. Before queueing, upload one image to `/ex-rvc/api/pushed-image`.
4. Queue the workflow.
5. The node consumes that uploaded image once.
6. The next queue run needs a new upload, unless `use_saved_image_id` or `use_comfy_upload_image` is selected explicitly.

## True Random Seed

Node display name:

```text
True Random Seed
```

Internal node key:

```text
TrueRandomSeed
```

This node outputs a 64-bit seed for sampler nodes. In `generate_new` mode it uses the operating system cryptographic random source through Python's `secrets` module. The seed is not derived from the previous seed, the workflow JSON, node IDs, or ComfyUI's `control_after_generate` behavior.

Outputs:

- `seed`: integer seed, range `0` to `18446744073709551615`
- `seed_text`: same seed as text, useful for logging or saving in metadata

### Seed Modes

`mode = generate_new`

Default mode. Generates a new 64-bit seed each time the node executes. This is intended for API workflows where the same workflow JSON is queued repeatedly and should still get a fresh unpredictable seed.

`mode = use_saved_seed`

Repro mode. The node returns the value from `saved_seed`. Use this when you want to re-run a previous result exactly.

### Typical Seed Flow

1. Add `True Random Seed`.
2. Keep `mode` set to `generate_new`.
3. Connect `seed` to a sampler seed input.
4. Save or inspect `seed_text` if you need to reproduce a run later.
5. For reproduction, set `mode` to `use_saved_seed` and paste the old seed into `saved_seed`.

---

## Beatdrop / Outfit Change Nodes

Category: `Amin/Beatdrop`

### AlphaRavis Judge (Same Thread)

Internal node key:

```text
AlphaRavisJudgeNode
```

Thin API client for AlphaRavis. It does not judge inside ComfyUI. It sends the selected frame URLs and metadata to the AlphaRavis OpenAI-compatible bridge so AlphaRavis can keep the current thread context and delegate the judgment to a subagent if needed.

Important inputs:

- `selected_paths`: newline-separated frame URLs from `DuoSelectorNode`.
- `alpha_endpoint`: usually `http://<AI_STACK_FIXED_IP>:8123/v1/chat/completions`.
- `conversation_id` / optional `thread_id`: required for same-thread routing.
- `job_policy`: `main_job_only`, `every_job`, `only_on_drop`, or `manual`.
- `run_id`, `job_id`, `drop_id`, `beats_json`, `context_json`: optional context for AlphaRavis.

Outputs:

- `verdict_json`: normalized AlphaRavis JSON verdict.
- `rejected_frames`: newline-separated rejected frame identifiers.
- `penalty_json`: map usable by `DuoSelectorNode.extra_penalty_json`.
- `should_restart`: boolean retry flag.
- `raw_response`: raw AlphaRavis answer text.

Typical loop:

```text
BeatItNode → FrameSequenceGenerator → DuoSelectorNode → AlphaRavisJudgeNode
                                      ↑                         │
                                      └──── penalty_json ───────┘
```

`DuoSelectorNode` always selects the lowest-penalty frames first, so rejected frames stay at the end on retries.

---

## Mask Researcher Tools

Custom nodes for video-analysis / researcher workflow.

### 1. Mask Quality Filter

Detect bad segmentation masks before interpolation. Catches: empty masks, tiny masks, sudden huge masks, too-small bounding boxes, centroid jumps, area jumps.

Use when SAM/SAM3/SAM3.1 sometimes tracks the wrong object.

Recommended pipeline:
```
SAM masks → Mask Quality Filter → Mask Interpolator Pro
```

Outputs: `filtered_masks`, `invalid_frame_mask`, `report`

### 2. Mask Interpolator Pro

Repair missing/empty masks over time. If SAM fails on frames, this fills the gap using optical-flow-guided or SDF interpolation.

Recommended pipeline:
```
Mask Quality Filter → Mask Interpolator Pro → Mask Crop Stabilizer
```

### 3. Mask Crop Stabilizer

Create stable crops from video frames + masks. Important before DINOv2/SigLIP/CLIP — unstable crops create false visual jumps.

Outputs: `crops`, `crop_masks`, `report`

### Why this matters for embedding models

If one frame has a broken mask or wildly different crop, DINOv2/SigLIP/CLIP may report a huge embedding difference (false positive). These nodes make the sequence stable before embedding comparison.

### Dependencies

Added to `requirements.txt`: `scipy`, `opencv-python-headless`
