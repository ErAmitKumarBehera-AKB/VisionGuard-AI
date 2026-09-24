from backend.app.services.decision_service import decide
def test_confidence_routes_low_confidence_to_review(monkeypatch):
    assert decide("DEFECT", 0.79).review_required is True
def test_confident_predictions_receive_automatic_final_label():
    assert decide("OK", 0.80).final_label == "OK"
    assert decide("DEFECT", 0.80).final_label == "DEFECT"
