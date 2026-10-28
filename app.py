"""OmniDoc RAG Chainlit web application entrypoint and RTL chat interface."""

import chainlit as cl

from src.common.logger import configure_logger, get_logger

logger = get_logger(__name__)


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize user chat session and display welcome guidance."""
    configure_logger(log_level="INFO", json_format=True)
    logger.info("chainlit_chat_session_started")

    welcome_content = (
        "سلام! به دستیار فنی **OmniDoc RAG** خوش آمدید.\n\n"
        "می‌توانید سوالات تخصصی خود درباره **Docker**، **PostgreSQL** یا **Linux** را "
        "به زبان فارسی یا انگلیسی مطرح کنید.\n\n"
        "> برای مثال:\n"
        "> `چگونه پورت‌های داکر را به صورت داینامیک نگاشت کنیم؟`"
    )

    await cl.Message(
        content=welcome_content,
        author="OmniDoc Assistant",
    ).send()  # type: ignore[no-untyped-call]


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle incoming user messages with bilingual direction and scaffolding.

    Note: Full streaming integration with LangGraph CRAG workflow, reasoning steps,
    and citations drawer is implemented in Task 9.2.
    """
    user_query = message.content.strip()
    logger.info("chainlit_message_received", query_len=len(user_query))

    response_text = (
        f"پیام شما با موفقیت دریافت شد:\n\n"
        f"**متن پرسش:** {user_query}\n\n"
        f"*(ارتباط کامل با موتور استریمینگ LangGraph CRAG در تسک ۹.۲ پیاده‌سازی خواهد شد.)*"
    )

    await cl.Message(
        content=response_text,
        author="OmniDoc Assistant",
    ).send()  # type: ignore[no-untyped-call]
