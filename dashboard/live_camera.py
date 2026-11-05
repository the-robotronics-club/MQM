"""Browser camera streaming with per-session, bounded inference state."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import av
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from inference.annotations import LEGEND, annotate_people
from inference.scene import analyze_scene

logger = logging.getLogger(__name__)


class LiveDetector:
    """Analyze at most two frames per second; retain only the latest result."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process_lock = threading.Lock()
        self.last_attempt = float("-inf")
        self.latest: Any = None
        self.state: dict[str, Any] = {}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def reset(self) -> None:
        with self.process_lock:
            with self.lock:
                self.state = {}
                self.latest = None
                self.last_attempt = float("-inf")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        with self.process_lock:
            now = time.monotonic()
            if now - self.last_attempt >= 0.5:
                self.last_attempt = now
                try:
                    from inference.model import detect_people

                    image = frame.to_ndarray(format="bgr24")
                    detections = detect_people(image)
                    posture_error = None
                    scene = None
                    try:
                        scene = analyze_scene(image, detections)
                    except Exception:
                        logger.exception("Live queue/posture analysis failed")
                        posture_error = (
                            "Queue and posture estimates are unavailable; "
                            "person counting continues."
                        )
                    annotated = annotate_people(
                        Image.fromarray(image[:, :, ::-1]),
                        scene["people"] if scene is not None else detections,
                    )
                    image = np.asarray(annotated)[:, :, ::-1].copy()
                    with self.lock:
                        self.latest = image
                        self.state = {
                            "headcount": len(detections),
                            "sitting": scene["sitting"] if scene is not None else None,
                            "standing": scene["standing"] if scene is not None else None,
                            "queue_count": scene["queue_count"] if scene is not None else None,
                            "standing_other_count": (
                                scene["standing_other_count"] if scene is not None else None
                            ),
                            "posture_available": scene is not None,
                            "posture_error": posture_error,
                            "seconds": time.monotonic() - now,
                            "updated": time.monotonic(),
                        }
                except Exception:
                    logger.exception("Live camera analysis failed")
                    with self.lock:
                        self.latest = None
                        self.state = {
                            "error": "Camera analysis failed. Stop and restart the camera."
                        }
            with self.lock:
                output = self.latest
            result = (
                av.VideoFrame.from_ndarray(output, format="bgr24") if output is not None else frame
            )
            result.pts = frame.pts
            result.time_base = frame.time_base
            return result


def render_live_camera() -> None:
    st.subheader("Phone camera")
    st.caption(
        "Tap START and allow camera access. Keep this page open to estimate the queue, "
        "sitting people, and other standing people continuously."
    )
    st.info("On your phone, open the HTTPS address. On this Mac, localhost also works.")
    st.caption(LEGEND)
    st.caption(
        "Video is sent to this Mac for analysis. Audio is off. Frames and results are not saved. "
        "Queue grouping uses people’s positions in the image and may confuse nearby groups."
    )
    if "live_posture_detector" not in st.session_state:
        st.session_state["live_posture_detector"] = LiveDetector()
    detector = st.session_state["live_posture_detector"]
    ctx = webrtc_streamer(
        key="phone-live-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": []},
        media_stream_constraints={
            "video": {
                "facingMode": {"ideal": "environment"},
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 5, "max": 10},
            },
            "audio": False,
        },
        video_frame_callback=detector.recv,
        on_video_ended=detector.reset,
        async_processing=True,
    )

    @st.fragment(run_every=1)
    def show_count() -> None:
        if not ctx.state.playing:
            st.caption("Camera stopped. Tap START to begin.")
            return
        state = detector.snapshot()
        if state.get("error"):
            st.error(state["error"])
        elif not state:
            st.info("Waiting for the first frame. Initial model loading can take a few seconds.")
        elif time.monotonic() - state["updated"] > 5:
            st.warning("No recent analysis. Check the camera connection or stop and restart.")
        else:
            with st.container(horizontal=True):
                st.metric("People", state["headcount"], border=True)
                st.metric(
                    "Queue estimate",
                    state.get("queue_count") if state.get("queue_count") is not None else "—",
                    border=True,
                )
                st.metric(
                    "Sitting",
                    state.get("sitting") if state.get("sitting") is not None else "—",
                    border=True,
                )
                st.metric(
                    "Other standing",
                    state.get("standing_other_count")
                    if state.get("standing_other_count") is not None
                    else "—",
                    border=True,
                )
            if state.get("posture_error"):
                st.warning(state["posture_error"])
            st.caption(f"Analysis: {state['seconds']:.2f}s · updates up to twice per second")

    show_count()
    st.caption("Use STOP to release the camera. Keep both devices on the same Wi-Fi.")
