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

Example response:

```json
{
  "ok": true,
  "available": true,
  "pending": true,
  "image_id": "6b4f7a9df5e24b289d0edc2f0e8c2d91",
  "filename": "6b4f7a9df5e24b289d0edc2f0e8c2d91.png",
  "size": 123456,
  "updated_at": "2026-05-22T12:00:00+00:00"
}
```

### Node Modes

`source_mode = require_new_api_push`

This is the default. The node loads the pending API image once and then clears the pending marker. The image file is kept for reproducibility, but it will not be reused automatically in the next queue run.

If no fresh image was pushed before queueing, the node blocks with an error instead of loading an old image.

`source_mode = use_saved_image_id`

Use this only when you intentionally want to reload a specific uploaded image. Paste the returned `image_id` into the node's `image_id` field.

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
6. The next queue run needs a new upload, unless `use_saved_image_id` is selected explicitly.
