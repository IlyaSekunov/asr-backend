import time
import httpx
import gradio as gr

API_BASE = "http://api:8000/api/v1/transcribe"
POLL_INTERVAL = 1.5
POLL_TIMEOUT = 300


def transcribe(audio_path: str) -> str:
    if audio_path is None:
        return "Загрузите или запишите аудиофайл."

    with httpx.Client(timeout=30) as client:
        # 1. Upload
        with open(audio_path, "rb") as f:
            filename = audio_path.split("/")[-1]
            resp = client.post(API_BASE + "/", files={"file": (filename, f)})
        resp.raise_for_status()
        task_id = resp.json()["task_id"]

        # 2. Poll
        deadline = time.time() + POLL_TIMEOUT
        while time.time() < deadline:
            time.sleep(POLL_INTERVAL)
            result = client.get(f"{API_BASE}/{task_id}")
            result.raise_for_status()
            data = result.json()

            if data["status"] == "READY":
                r = data["result"]
                return (
                    f"{r['text']}\n\n"
                    f"Язык: {r['language']} "
                    f"(уверенность {r['language_probability']:.0%})"
                )
            if data["status"] == "FAILED":
                return "Ошибка транскрибации на стороне сервера."

    return "Превышено время ожидания ответа."


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="Аудио"),
    outputs=gr.Textbox(label="Транскрипция", lines=6),
    title="Speech Processing API",
    description="Загрузите .mp3/.wav или запишите голос с микрофона.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
