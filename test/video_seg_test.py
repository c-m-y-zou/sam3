
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from sam3.model_builder import build_sam3_video_predictor

# video_path needs to be either a JPEG folder or an MP4 video file
video_path = "/home/y-zou/kajima_sam3/sam3/test_data/360 Movie.MP4"
video_id = Path(video_path).stem.split("_")[0]
out_dir = Path("/home/y-zou/kajima_sam3/sam3/test_data")

# use all available GPUs on the machine
gpus_to_use = range(torch.cuda.device_count())

# Load the model
predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
session_id = None

cap = cv2.VideoCapture(video_path)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# a 1408-frame, high-res fisheye video grows the tracked-object memory bank
# until it OOMs if propagated in one shot. Chunking + a full close/restart per
# chunk (see below) rules out a cross-chunk leak: it still OOM'd at the same
# spot (frame ~855) even with a completely fresh session for that chunk, so
# that stretch of video alone makes the generic "metal clamp" prompt latch
# onto enough distinct objects to exhaust 47GB within one chunk. Mitigate with
# a smaller chunk size and a stricter confidence threshold (fewer borderline
# detections promoted to persistently-tracked objects).
CHUNK_FRAMES = 200
OUTPUT_PROB_THRESH = 0.6

colors_bgr = [
    tuple(int(c * 255) for c in color[::-1])  # matplotlib RGB -> OpenCV BGR
    for color in matplotlib.colormaps["tab10"].colors
]

# high-res fisheye frames need thicker lines / bigger text than a typical photo
scale = max(frame_h, frame_w) / 1500.0
scale = max(scale, 1.0)
box_thickness = max(1, round(3 * scale))
font_scale = 0.6 * scale
text_thickness = max(1, round(1.5 * scale))


def draw_detections(frame_bgr, out):
    for i, obj_id in enumerate(out["out_obj_ids"]):
        color = colors_bgr[obj_id % len(colors_bgr)]
        mask = out["out_binary_masks"][i].astype(bool)
        frame_bgr[mask] = (0.4 * frame_bgr[mask] + 0.6 * np.array(color)).astype(np.uint8)

        x, y, w, h = out["out_boxes_xywh"][i] * [frame_w, frame_h, frame_w, frame_h]
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, box_thickness)

        label = f"id={obj_id} score={out['out_probs'][i]:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        cv2.rectangle(frame_bgr, (x, y - th - baseline - 4), (x + tw + 4, y), color, -1)
        cv2.putText(
            frame_bgr,
            label,
            (x + 2, y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    return frame_bgr


def get_frame(frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    return frame_bgr if ok else None


# Prompts to compare side by side
PROMPTS = [
    # "metal clamp",
    "green clamp on steel beam",
]

for prompt in PROMPTS:
    slug = prompt.replace(" ", "_")
    out_path = out_dir / f"result_{video_id}_{slug}.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h)
    )

    for chunk_start in range(0, num_frames, CHUNK_FRAMES):
        chunk_len = min(CHUNK_FRAMES, num_frames - chunk_start)

        # fully close the previous chunk's session and start a fresh one so
        # its GPU memory (tracked objects, memory bank, ...) is actually
        # released, rather than just dropping Python references
        if session_id is not None:
            predictor.handle_request(request=dict(type="close_session", session_id=session_id))
        response = predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path)
        )
        session_id = response["session_id"]

        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=chunk_start,
                text=prompt,
                output_prob_thresh=OUTPUT_PROB_THRESH,
            )
        )

        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="forward",
                start_frame_index=chunk_start,
                max_frame_num_to_track=chunk_len,
                output_prob_thresh=OUTPUT_PROB_THRESH,
            )
        ):
            frame_idx, out = response["frame_index"], response["outputs"]
            frame_bgr = get_frame(frame_idx)
            if frame_bgr is None:
                continue
            draw_detections(frame_bgr, out)
            writer.write(frame_bgr)

        print(f"[{prompt!r}] processed frames {chunk_start}-{chunk_start + chunk_len - 1}")

    writer.release()
    print(f"saved video to {out_path}")

cap.release()
predictor.handle_request(request=dict(type="close_session", session_id=session_id))
predictor.shutdown()
