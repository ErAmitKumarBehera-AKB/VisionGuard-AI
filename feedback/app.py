import pandas as pd
import streamlit as st

from qc.components.inspection_viewer import render_inspection_review
from qc.components.metrics_card import render_metrics_cards
from qc.components.sidebar import render_sidebar
from qc.services.api_client import QCBackendClient

st.set_page_config(
    page_title="HITL Quality Control - Visual Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

client = QCBackendClient()


def main():
    st.title("Human-in-the-Loop Quality Control Dashboard")
    st.caption("Industrial Visual Defect Verification & Continuous Retraining Platform")

    filters = render_sidebar()

    stats = client.get_summary_stats()
    render_metrics_cards(stats)
    st.markdown("---")

    tab_inspect, tab_retrain, tab_docs = st.tabs([
        "📋 Live Inspections & Verification",
        "🔄 Retraining Pool (DVC)",
        "📖 Inspection Guidelines",
    ])

    with tab_inspect:
        history_data = client.get_history(
            page=1,
            page_size=50,
            prediction=filters["prediction"],
            product_category=filters["product_category"],
            low_confidence_only=filters["low_confidence_only"],
        )
        items = history_data.get("items", [])

        if not items:
            st.info("No inspection records found matching the active filters.")
        else:
            col_list, col_review = st.columns([1, 1])

            with col_list:
                st.subheader(f"Inspections ({len(items)} records)")

                records_list = []
                for item in items:
                    records_list.append({
                        "ID": item["inspection_id"],
                        "Prediction": item["prediction"],
                        "Confidence": f"{item['confidence'] * 100:.1f}%",
                        "Category": item.get("product_category", "-"),
                        "Latency": f"{item.get('latency_ms', 0):.1f}ms",
                        "Status": item.get("feedback_status", "PENDING"),
                    })

                df_display = pd.DataFrame(records_list)
                st.dataframe(df_display, use_container_width=True, height=450)

                selected_id = st.selectbox(
                    "Select Inspection to Inspect & Label:",
                    options=[item["inspection_id"] for item in items],
                    index=0,
                )

            with col_review:
                selected_item = next((i for i in items if i["inspection_id"] == selected_id), None)
                if selected_item:
                    def handle_feedback(insp_id: str, label: str, notes: str):
                        client.submit_feedback(insp_id, label, notes)

                    render_inspection_review(selected_item, on_submit_feedback=handle_feedback)

    with tab_retrain:
        st.subheader("Validated Human Feedback Pool for Retraining")
        st.write(
            "This table contains corrections and verifications submitted by human inspectors. "
            "When the pool reaches the minimum batch threshold, these samples are ingested "
            "into the unified dataset manifest, versioned with **DVC**, and used for model fine-tuning."
        )

        fb_export = client.get_feedback_export()
        feedback_items = fb_export.get("items", [])

        if feedback_items:
            st.success(f"{len(feedback_items)} validated samples ready for DVC versioning and retraining.")
            st.dataframe(pd.DataFrame(feedback_items), use_container_width=True)

            if st.button("Trigger Retraining Pipeline Check"):
                st.info(
                    "Triggering retraining workflow...\n"
                    "Command: `python training/scripts/prepare_dataset.py && python training/scripts/train.py`"
                )
        else:
            st.info("No validated human feedback submitted yet. Review inspections above to populate.")

    with tab_docs:
        st.subheader("Quality Control Standard Operating Procedure (SOP)")
        st.markdown("""
        ### Zero-Defect Inspection Protocol
        1. **Low Confidence Triage**: Any part predicted with `< 80%` confidence must undergo mandatory visual inspection by a human QC specialist.
        2. **False Alarm vs True Defect**:
           - **False Alarm (False Positive)**: If cosmetic dust or glare caused the model to predict `DEFECT`, correct the label to `OK` and note *'glare/lighting issue'*.
           - **Missed Defect (False Negative)**: If a real hairline crack or bent pin was predicted as `OK`, immediately correct the label to `DEFECT`. This sample will be assigned high weight during next training.
        3. **DVC Pipeline**: Human feedback is committed to Git/DVC to ensure data lineage and model auditability for ISO 9001 compliance.
        """)


if __name__ == "__main__":
    main()
