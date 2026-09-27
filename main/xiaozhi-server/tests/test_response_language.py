"""Spanish conversations must not inherit Chinese examples or display mixed output."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from core.connection import ConnectionHandler, DIRECT_ANSWER_TOOL, direct_answer_tool
from core.providers.llm.ollama.ollama import LLMProvider
from core.providers.tools.device_mcp.mcp_executor import DeviceMCPExecutor
from core.providers.tts.dto.dto import SentenceType
from core.providers.tts.text_only import TTSProvider
from core.utils.dialogue import Dialogue
from core.utils.prompt_manager import PromptManager
from core.utils.response_language import (
    HAN_CHARACTERS, SPANISH_ERROR_RESPONSE, prepare_spanish_response, uses_spanish,
)
from plugins_func.register import Action


class ResponseLanguageTests(unittest.TestCase):
    def connection(self):
        return SimpleNamespace(
            config={"response_language": "es"}, session_id="session-1", llm=MagicMock(),
        )

    def test_spanish_text_does_not_make_an_extra_model_request(self):
        conn = self.connection()
        text = "Grand Theft Auto es un juego de acción y aventura."
        self.assertEqual(prepare_spanish_response(text, conn), text)
        conn.llm.response.assert_not_called()

    def test_mixed_sentence_is_repaired_without_history_or_tools(self):
        conn = self.connection()
        stream = MagicMock()
        stream.__iter__.return_value = iter(["¿Puedes dar ", "más detalles?"])
        conn.llm.response.return_value = stream
        original = "Grand Theft Auto es un juego donde你能提供更详细的描述吗？"
        result = prepare_spanish_response(original, conn)
        self.assertEqual(result, "Grand Theft Auto es un juego donde ¿Puedes dar más detalles?")
        args, kwargs = conn.llm.response.call_args
        self.assertEqual(args[0], "session-1")
        self.assertEqual([item["role"] for item in args[1]], ["system", "user"])
        self.assertEqual(args[1][1]["content"], "你能提供更详细的描述吗？")
        self.assertEqual(kwargs, {"temperature": 0, "max_tokens": 384, "timeout": 10})
        stream.close.assert_called_once()

    def test_each_chinese_fragment_is_repaired_and_spanish_is_preserved(self):
        conn = self.connection()
        conn.llm.response.side_effect = [iter(["azul"]), iter(["verde"])]
        self.assertEqual(prepare_spanish_response("Uno: 蓝色. Dos: 绿色. Fin.", conn), "Uno: azul. Dos: verde. Fin.")
        self.assertEqual(conn.llm.response.call_count, 2)

    def test_failed_empty_and_still_chinese_repairs_never_publish_chinese(self):
        for output in ([], ["还是中文"], ["   "]):
            with self.subTest(output=output):
                conn = self.connection()
                conn.config["system_error_response"] = "中文错误"
                conn.llm.response.return_value = iter(output)
                self.assertEqual(prepare_spanish_response("Hola世界", conn), SPANISH_ERROR_RESPONSE)
        conn = self.connection()
        conn.config["system_error_response"] = "Inténtalo otra vez."
        conn.llm.response.side_effect = TimeoutError()
        self.assertEqual(prepare_spanish_response("Hola世界", conn), "Inténtalo otra vez.")

    def test_display_queue_preserves_turn_and_emits_no_audio_after_repair(self):
        provider = TTSProvider({}, True)
        provider.conn = self.connection()
        provider.conn.llm.response.return_value = iter(["¿Puedes dar más detalles?"])
        provider.current_sentence_id = "turn-1"
        audio_handler = MagicMock()
        provider.to_tts_stream("Un juego. 你能提供更详细的描述吗？", audio_handler)
        self.assertEqual(provider.tts_audio_queue.get_nowait(), (
            SentenceType.FIRST, [], "Un juego. ¿Puedes dar más detalles?", "turn-1",
        ))
        audio_handler.assert_not_called()

    def test_provider_sends_repair_limits_to_ollama(self):
        with patch("core.providers.llm.ollama.ollama.OpenAI"):
            provider = LLMProvider({"model_name": "qwen2.5:7b"})
        stream = MagicMock()
        stream.__iter__.return_value = iter([SimpleNamespace(choices=[
            SimpleNamespace(delta=SimpleNamespace(content="Hola")),
        ])])
        provider.client.chat.completions.create.return_value = stream
        self.assertEqual(list(provider.response(
            "session", [], temperature=0, max_tokens=384, timeout=10,
        )), ["Hola"])
        kwargs = provider.client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0)
        self.assertEqual(kwargs["max_tokens"], 384)
        self.assertEqual(kwargs["timeout"], 10)
        stream.close.assert_called_once()

    def test_spanish_locale_and_existing_text_only_config_are_detected(self):
        for language in ("es", "es-CL", "es_ES", "español", "spanish"):
            self.assertTrue(uses_spanish({"response_language": language}))
        self.assertTrue(uses_spanish({
            "selected_module": {"TTS": "TextOnly"},
            "TTS": {"TextOnly": {"language": "español"}},
        }))
        self.assertFalse(uses_spanish({"response_language": "zh"}))
        self.assertFalse(uses_spanish({}))

    def test_examples_keep_the_tool_protocol_and_spanish_language(self):
        conn = self.connection()
        conn.intent_type = "function_call"
        conn.dialogue = Dialogue()
        conn.logger = MagicMock()
        conn.func_handler = SimpleNamespace(get_functions=lambda: [
            {"function": {"name": "handle_exit_intent"}},
        ])
        conn.dialogue.update_system_message("Responde exclusivamente en español.")
        ConnectionHandler._inject_tool_call_fewshot(conn)
        messages = conn.dialogue.get_llm_dialogue()
        self.assertIsNone(HAN_CHARACTERS.search(json.dumps(messages, ensure_ascii=False)))
        pending = {}
        for item in messages:
            for call in item.get("tool_calls", []):
                pending[call["id"]] = call
                json.loads(call["function"]["arguments"])
            if item["role"] == "tool":
                self.assertIn(item["tool_call_id"], pending)
                del pending[item["tool_call_id"]]
        self.assertEqual(pending, {})
        self.assertEqual(len(messages), 8)

    def test_spanish_tool_description_does_not_change_other_languages(self):
        before = json.dumps(DIRECT_ANSWER_TOOL)
        spanish = direct_answer_tool({"response_language": "es"})
        self.assertIsNone(HAN_CHARACTERS.search(json.dumps(spanish, ensure_ascii=False)))
        self.assertEqual(json.dumps(DIRECT_ANSWER_TOOL), before)
        self.assertEqual(direct_answer_tool({}), DIRECT_ANSWER_TOOL)

    def test_weekday_in_spanish_prompt_is_spanish(self):
        with patch("core.utils.current_time.get_current_weekday", return_value="星期日"):
            spanish = PromptManager({"response_language": "es"}, MagicMock())
            legacy = PromptManager({}, MagicMock())
            self.assertEqual(spanish._get_current_time_info()[1], "domingo")
            self.assertEqual(legacy._get_current_time_info()[1], "星期日")


class DeviceErrorLanguageTests(unittest.IsolatedAsyncioTestCase):
    async def test_camera_failure_is_shown_in_spanish(self):
        conn = SimpleNamespace(
            config={"response_language": "es"},
            mcp_client=SimpleNamespace(is_ready=AsyncMock(return_value=True)),
        )
        executor = DeviceMCPExecutor(conn)
        with patch("core.providers.tools.device_mcp.mcp_executor.call_mcp_tool", new=AsyncMock(
            side_effect=Exception("MCP错误: Failed to capture photo"),
        )):
            result = await executor.execute(conn, "self_camera_take_photo", {})
        self.assertEqual(result.action, Action.ERROR)
        self.assertEqual(result.response, "No pude tomar la foto. Revisa la cámara e inténtalo otra vez.")

    async def test_disconnected_device_returns_spanish_without_model_request(self):
        conn = SimpleNamespace(config={"response_language": "es"})
        result = await DeviceMCPExecutor(conn).execute(conn, "self_camera_take_photo", {})
        self.assertEqual(result.action, Action.ERROR)
        self.assertIsNone(HAN_CHARACTERS.search(result.response))
        self.assertIn("placa", result.response)

    async def test_async_device_timeout_returns_spanish(self):
        conn = SimpleNamespace(
            config={"response_language": "es"},
            mcp_client=SimpleNamespace(is_ready=AsyncMock(return_value=True)),
        )
        with patch("core.providers.tools.device_mcp.mcp_executor.call_mcp_tool", new=AsyncMock(
            side_effect=asyncio.TimeoutError(),
        )):
            result = await DeviceMCPExecutor(conn).execute(conn, "self_camera_take_photo", {})
        self.assertEqual(result.response, "La placa no respondió a tiempo. Inténtalo otra vez.")


if __name__ == "__main__":
    unittest.main()
