from emotion_model import predict_emotions


def test_predict_emotions_detects_joy():
    result = predict_emotions("I feel happy and grateful today.")
    assert result["joy"] > result["sadness"]
    assert result["joy"] > 0.0


def test_predict_emotions_detects_fear_and_anger():
    result = predict_emotions("I am terrified and furious after the attack.")
    assert result["fear"] > 0.0
    assert result["anger"] > 0.0


def test_predict_emotions_handles_clean_text():
    result = predict_emotions("The sun is shining.")
    assert sum(result.values()) >= 0.0
