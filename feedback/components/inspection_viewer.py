from pathlib import Path
from typing import Any, Callable
from PIL import Image
import streamlit as st


def render_inspection_review(
    inspection: dict[str, Any],
    on_submit_feedback: Callable[[str, str, str], None],
) -> None:
    st.subheader(f"Reviewing Inspection: `{inspection['inspection_id']}`")

    col_img, col_meta = st.columns([1, 1])

    with col_img:
        img_ref = inspection.get("image_reference")
        if img_ref and Path(img_ref).is_file():
            img = Image.open(img_ref)
            st.image(img, caption=f"Sample: {inspection.get('product_category', 'Part')}", use_container_width=True)
        else:
            st.info("Sample image not saved on disk or placeholder.")

    with col_meta:
        pred = inspection.get("prediction", "UNKNOWN")
        confidence = float(inspection.get("confidence", 0.0))
        status = inspection.get("feedback_status", "PENDING")

        if pred == "DEFECT":
            st.error(f"Predicted: **DEFECT** ({confidence * 100:.1f}% Confidence)")
        else:
            st.success(f"Predicted: **OK** ({confidence * 100:.1f}% Confidence)")

        st.progress(min(max(confidence, 0.0), 1.0))

        st.write(f"**Product Category:** {inspection.get('product_category', 'N/A')}")
        st.write(f"**Model Version:** `{inspection.get('model_version', 'v1.0.0')}`")
        st.write(f"**Inference Latency:** {inspection.get('latency_ms', 0.0):.1f} ms")
        st.write(f"**Review Status:** `{status}`")
        if inspection.get("operator_label"):
            st.write(f"**Current Operator Label:** `{inspection.get('operator_label')}`")

        st.markdown("---")
        st.markdown("#### Operator Correction")

        default_idx = 1 if pred == "DEFECT" else 0
        corrected_label = st.radio(
            "Verify or Correct Ground Truth Label:",
            options=["OK", "DEFECT"],
            index=default_idx,
            horizontal=True,
            key=f"radio_{inspection['inspection_id']}",
        )

        comments = st.text_input(
            "QC Inspector Notes / Defect Type:",
            placeholder="e.g., surface scratch, bent pin, false alarm caused by lighting",
            key=f"notes_{inspection['inspection_id']}",
        )

        if st.button("Submit Label to Retraining Pool", type="primary", key=f"btn_{inspection['inspection_id']}"):
            on_submit_feedback(inspection["inspection_id"], corrected_label, comments)
            st.success(f"Feedback recorded: Confirmed as '{corrected_label}'!")
            st.rerun()
