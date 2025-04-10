import gradio as gr
import requests
import re

HF_TOKEN = ""
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

TRIGGER_KEYWORDS = {
    "cancel": "Customer threatening to cancel — consider escalation.",
    "useless": "Customer feels unsupported — suggest offering solutions.",
    "angry": "Customer is emotional — acknowledge frustration.",
    "not working": "System issue — show urgency and reassurance.",
    "called twice": "Repeated attempts — apologize and take ownership."
}

SENTIMENT_THRESHOLD = 0.85

def query_sentiment(text):
    url = f"https://api-inference.huggingface.co/models/{SENTIMENT_MODEL}"
    response = requests.post(url, headers=HEADERS, json={"inputs": text})
    try:
        result = response.json()
        first = result[0] if isinstance(result, list) else result
        label = first.get("label", "")
        score = round(first.get("score", 0), 3)
        return label, score
    except:
        return "UNKNOWN", 0

def check_alert(input_text):
    if ":" not in input_text:
        return {"error": "Please format as 'Speaker: message'"}

    speaker, text = input_text.split(":", 1)
    speaker = speaker.strip()
    message = text.strip().lower()

    label, score = query_sentiment(message)

    alert_triggered = False
    reason = None
    tip = None

    # Keyword check
    for kw, advice in TRIGGER_KEYWORDS.items():
        if re.search(rf"\b{re.escape(kw)}\b", message):
            alert_triggered = True
            reason = f"Keyword match: {kw}"
            tip = advice
            break

    # Sentiment check
    if not alert_triggered and "1" in label and score >= SENTIMENT_THRESHOLD:
        alert_triggered = True
        reason = "High negative sentiment"
        tip = "Acknowledge the customer's frustration and offer a resolution."

    return {
        "alert": alert_triggered,
        "trigger": reason if alert_triggered else "None",
        "speaker": speaker,
        "text": text.strip(),
        "sentiment": f"{label} ({score})",
        "coaching_tip": tip if alert_triggered else "No coaching needed."
    }

# Gradio UI
gr.Interface(
    fn=check_alert,
    inputs=gr.Textbox(label="Enter Turn (e.g., Customer: I'm very angry!)"),
    outputs="json",
    title="📞 Real-Time Agent Alert Engine",
    description="Detects customer frustration and suggests coaching actions"
).launch(share=True)
