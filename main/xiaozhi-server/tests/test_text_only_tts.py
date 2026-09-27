"""Text responses must reach the existing display protocol without TTS audio."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from core.handle.sendAudioHandle import sendAudioMessage
from core.providers.tts.dto.dto import SentenceType
from core.providers.tts.text_only import TTSProvider
from core.utils.modules_initialize import initialize_tts


class TextOnlyTtsTests(unittest.TestCase):
    def test_selected_provider_can_be_initialized(self):
        provider = initialize_tts(
            {
                "selected_module": {"TTS": "TextOnly"},
                "TTS": {"TextOnly": {"type": "text_only", "language": "español"}},
                "delete_audio": True,
            }
        )
        self.assertIsInstance(provider, TTSProvider)

    def test_preserves_sentence_text_and_turn_identity_without_audio(self):
        provider = TTSProvider({}, True)
        provider.current_sentence_id = "turn-1"
        audio_handler = MagicMock()
        provider.to_tts_stream("¡Hola! Puedo leer imágenes.", audio_handler)
        self.assertEqual(
            provider.tts_audio_queue.get_nowait(),
            (SentenceType.FIRST, [], "¡Hola! Puedo leer imágenes.", "turn-1"),
        )
        audio_handler.assert_not_called()

    def test_empty_text_emits_no_sentence(self):
        provider = TTSProvider({}, True)
        provider.to_tts_stream("")
        self.assertTrue(provider.tts_audio_queue.empty())

    def test_direct_calls_produce_no_audio_file(self):
        provider = TTSProvider({}, False)
        with patch("builtins.open", side_effect=AssertionError("No files expected")):
            self.assertEqual(provider.to_tts("Hola"), [])
            self.assertEqual(asyncio.run(provider.text_to_speak("Hola", "unused.wav")), b"")

    def test_real_sender_delivers_text_and_stop_without_binary_packets(self):
        async def run():
            provider = TTSProvider({}, True)
            provider.current_sentence_id = "turn-1"
            provider.to_tts_stream("Esta respuesta aparece en la página.")
            conn = SimpleNamespace(
                tts=provider,
                sentence_id="turn-1",
                session_id="session-1",
                logger=MagicMock(),
                websocket=SimpleNamespace(send=AsyncMock()),
                calling=False,
                close_after_chat=False,
                config={"enable_stop_tts_notify": False},
                clearSpeakStatus=MagicMock(),
            )
            await sendAudioMessage(conn, *provider.tts_audio_queue.get_nowait())
            await sendAudioMessage(conn, SentenceType.LAST, [], None, "turn-1")
            messages = [json.loads(call.args[0]) for call in conn.websocket.send.call_args_list]
            self.assertEqual([message["state"] for message in messages], ["sentence_start", "stop"])
            self.assertEqual(messages[0]["text"], "Esta respuesta aparece en la página.")
            conn.clearSpeakStatus.assert_called_once()

        asyncio.run(run())

    def test_interrupted_old_turn_does_not_publish_text(self):
        async def run():
            provider = TTSProvider({}, True)
            conn = SimpleNamespace(sentence_id="new-turn", tts=provider)
            await sendAudioMessage(conn, SentenceType.FIRST, [], "Respuesta vieja", "old-turn")
            self.assertTrue(provider.tts_audio_first_sentence)

        asyncio.run(run())
