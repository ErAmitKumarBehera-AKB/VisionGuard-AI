from typing import Any
import streamlit as st


def render_metrics_cards(stats: dict[str, Any]) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Inspected",
            value=f"{stats.get('total_inspections', 0):,}",
        )
    with col2:
        defect_rate = stats.get("defect_rate_percentage", 0.0)
        st.metric(
            label="Defect Rate",
            value=f"{defect_rate:.1f}%",
            delta=f"{stats.get('defect_count', 0)} defects",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            label="Avg Latency",
            value=f"{stats.get('average_latency_ms', 0.0):.1f} ms",
        )
    with col4:
        low_conf = stats.get("low_confidence_count", 0)
        st.metric(
            label="Low Confidence Alert",
            value=f"{low_conf}",
            delta="Review Needed" if low_conf > 0 else "Optimal",
            delta_color="off" if low_conf == 0 else "inverse",
        )
    with col5:
        reviewed = stats.get("reviewed_count", 0)
        st.metric(
            label="Operator Reviewed",
            value=f"{reviewed}",
        )
