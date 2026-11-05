"""Camera-only entry point for temporary HTTPS access from a phone."""

import hmac
import os

import streamlit as st

st.set_page_config(page_title="MQM phone camera", page_icon=":material/videocam:")
st.title("MQM phone camera")
expected = os.environ.get("PHONE_CAMERA_ACCESS_CODE", "")
if not expected:
    st.error("Set PHONE_CAMERA_ACCESS_CODE before starting the phone camera page.")
    st.stop()
if not st.session_state.get("phone_authorized"):
    with st.form("phone_access"):
        code = st.text_input("Access code", type="password")
        submitted = st.form_submit_button("Connect")
    if submitted:
        if hmac.compare_digest(code.encode(), expected.encode()):
            st.session_state["phone_authorized"] = True
            st.rerun()
        else:
            st.error("Incorrect access code.")
    st.stop()

from dashboard.live_camera import render_live_camera  # noqa: E402

render_live_camera()
