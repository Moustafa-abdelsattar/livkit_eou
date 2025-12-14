"""
Arabic Voice Agent with Turn Detection

LiveKit voice agent with fine-tuned Arabic turn detector.
"""

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession
from livekit.plugins import groq, silero
from livekit_plugins_arabic_turn_detector import load

load_dotenv(".env.local")

print("="*60)
print("Testing Arabic Turn Detector SDK")
print("="*60)


class ArabicAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""أنت مساعد صوتي ذكي باللغة العربية.
تحدث دائماً باللغة العربية الفصحى أو اللهجة الخليجية حسب المستخدم.
أنت مفيد وودود وتجيب على الأسئلة بوضوح.
إجاباتك قصيرة ومباشرة - جملة أو جملتين فقط.
لا تستخدم تنسيق معقد أو رموز تعبيرية.
استمع جيداً واستجب بسرعة عندما ينهي المستخدم كلامه.""",
        )


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    print("\nLoading Arabic Turn Detector from SDK...")

    session = AgentSession(
        stt=groq.STT(language="ar"),
        llm="openai/gpt-4o",
        tts="cartesia/sonic-3",
        # Using YOUR Arabic Turn Detector SDK!
        # TEMPORARY: Using higher threshold due to model over-predicting
        turn_detection=load(threshold=0.98),
        vad=silero.VAD.load(),
        allow_interruptions=True,
    )

    print("Arabic Turn Detector loaded successfully!")
    print("SDK is working! Your fine-tuned model is active.\n")

    await session.start(
        room=ctx.room,
        agent=ArabicAssistant(),
    )

    await session.generate_reply(
        instructions="رحب بالمستخدم باللغة العربية واعرض مساعدتك."
    )


if __name__ == "__main__":
    print("\n[OK] SDK imported successfully!")
    print("[OK] Starting agent with Arabic Turn Detector...\n")
    agents.cli.run_app(server)
