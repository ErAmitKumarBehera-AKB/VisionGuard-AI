from typing import Any
import streamlit as st


def render_sidebar() -> dict[str, Any]:
    st.sidebar.title("Inspection Filters")
    st.sidebar.markdown("---")

    prediction_filter = st.sidebar.selectbox(
        "Prediction Label",
        options=["ALL", "DEFECT", "OK"],
        index=0,
        help="Filter inspections by model classification outcome.",
    )

    category_filter = st.sidebar.selectbox(
        "Product Category",
        options=["ALL", "cable", "screw", "metal_nut", "transistor", "casting_impeller"],
        index=0,
    )

    low_conf_only = st.sidebar.checkbox(
        "Low Confidence Flagged Only (< 80%)",
        value=False,
        help="Prioritize samples with uncertain classifications for immediate human verification.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Info")
    st.sidebar.info("Model: ResNet-50 Transfer Learning\nTarget: Industrial Zero-Defect Quality")

    return {
        "prediction": prediction_filter,
        "product_category": category_filter,
        "low_confidence_only": low_conf_only,
    }
