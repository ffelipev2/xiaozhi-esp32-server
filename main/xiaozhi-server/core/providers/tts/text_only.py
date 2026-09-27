"""Deliver assistant sentences to the device display without speech synthesis."""

from core.providers.tts.base import TTSProviderBase
from core.providers.tts.dto.dto import SentenceType
from core.utils.response_language import prepare_spanish_response, uses_spanish


class TTSProvider(TTSProviderBase):
    def to_tts_stream(self, text, opus_handler=None):
        if text:
            if self.conn is not None and uses_spanish(self.conn.config):
                text = prepare_spanish_response(text, self.conn)
            # Keep the existing subtitle and turn-completion protocol. The base
            # worker sends LAST/stop when the LLM finishes, with no audio packets.
            self.tts_audio_queue.put(
                (SentenceType.FIRST, [], text, getattr(self, "current_sentence_id", None))
            )

    def to_tts(self, text):
        return []

    async def text_to_speak(self, text, output_file):
        return b""
