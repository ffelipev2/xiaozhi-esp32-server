"""Camera routing must use advertised tools and preserve the MCP call contract."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from core.providers.llm.ollama.ollama import LLMProvider
from core.utils.camera_requests import (
    CAMERA_TOOL, camera_question, is_camera_request, wait_for_camera_registration,
)


CAMERA = {"type": "function", "function": {"name": CAMERA_TOOL}}


class CameraRegistrationTests(unittest.IsolatedAsyncioTestCase):
    def connection(self, ready=False):
        return SimpleNamespace(
            llm=SimpleNamespace(route_camera_requests=True), client_abort=False,
            mcp_client=SimpleNamespace(is_ready=AsyncMock(return_value=ready)),
        )

    async def test_first_request_waits_without_blocking_mcp_replies(self):
        conn = self.connection()

        async def register():
            await asyncio.sleep(0.01)
            conn.mcp_client.is_ready.return_value = True

        registration = asyncio.create_task(register())
        await asyncio.wait_for(wait_for_camera_registration(conn, "Toma una foto"), 1)
        await registration
        self.assertGreater(conn.mcp_client.is_ready.await_count, 1)

    async def test_missing_reply_has_bounded_wait(self):
        conn = self.connection()
        await asyncio.wait_for(
            wait_for_camera_registration(conn, "Toma una foto", timeout=0.01), 0.5
        )
        self.assertFalse(conn.mcp_client.is_ready.return_value)

    async def test_ready_absent_disabled_and_unrelated_requests_do_not_wait(self):
        ready = self.connection(ready=True)
        await wait_for_camera_registration(ready, "Toma una foto")
        ready.mcp_client.is_ready.assert_awaited_once()
        for query, disabled, absent in (
            ("¿Cuánto es dos más dos?", False, False),
            ("Toma una foto", True, False),
            ("Toma una foto", False, True),
        ):
            conn = self.connection()
            client = conn.mcp_client
            conn.llm.route_camera_requests = not disabled
            if absent:
                del conn.mcp_client
            await wait_for_camera_registration(conn, query)
            client.is_ready.assert_not_awaited()


class CameraRequestTests(unittest.TestCase):
    def test_explicit_visual_requests_and_observed_asr_error(self):
        for query in (
            "¿Qué ves?", "¿Qué es lo que ves?", "quién lo que ves",
            "Describe lo que ves", "Toma una foto", "Por favor, mira esto.",
            "Usa la cámara", "Activa la cámara",
            "Lee el texto que tengo enfrente", "¿Qué tengo delante?",
        ):
            with self.subTest(query=query):
                self.assertTrue(is_camera_request(query))

    def test_other_questions_and_negations_do_not_capture(self):
        for query in (
            "¿Puedes usar la cámara?", "¿Tienes cámara?", "¿Cuánto es dos más dos?",
            "No tomes una foto", "Describe lo que ves sin usar la cámara",
            "Cuéntame una historia sobre una cámara", "Lee un cuento",
            "Explica qué ves significa en español", "¿Qué hay de nuevo?", "¿Qué hay?", None,
        ):
            with self.subTest(query=query):
                self.assertFalse(is_camera_request(query))

    def test_capture_commands_ask_vision_to_describe_the_received_photo(self):
        for query in ("Toma una foto", "Por favor, usa la cámara", "Saca una foto", "Tómame una foto"):
            with self.subTest(query=query):
                self.assertEqual(
                    camera_question(query),
                    "Describe brevemente lo que ves en esta imagen.",
                )
        for query in ("¿Qué ves?", "Lee el texto que tengo enfrente", "No tomes una foto"):
            self.assertEqual(camera_question(query), query)

    def test_capture_tool_receives_an_analysis_question(self):
        provider = self.make_provider()
        _, calls = list(provider.response_with_functions(
            "test-session", [{"role": "user", "content": "Toma una foto"}], [CAMERA],
        ))[0]
        self.assertEqual(json.loads(calls[0].function.arguments), {
            "question": "Describe brevemente lo que ves en esta imagen."
        })

    def make_provider(self, enabled=True):
        with patch("core.providers.llm.ollama.ollama.OpenAI") as factory:
            provider = LLMProvider({
                "model_name": "qwen2.5:7b", "route_camera_requests": enabled,
            })
        stream = MagicMock()
        stream.__iter__.return_value = iter([
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                content="Respuesta del modelo.", tool_calls=None,
            ))])
        ])
        factory.return_value.chat.completions.create.return_value = stream
        return provider

    def test_routes_once_with_original_question_and_native_call_fields(self):
        provider = self.make_provider()
        query = "¿Qué ves?"
        result = list(provider.response_with_functions(
            "test-session", [{"role": "user", "content": query}], [CAMERA],
        ))
        self.assertEqual(len(result), 1)
        content, calls = result[0]
        self.assertIsNone(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, CAMERA_TOOL)
        self.assertEqual(json.loads(calls[0].function.arguments), {"question": query})
        self.assertEqual(calls[0].index, 0)
        self.assertTrue(calls[0].id)
        provider.client.chat.completions.create.assert_not_called()

    def test_unavailable_camera_disabled_routing_and_tool_results_use_llm(self):
        scenarios = (
            (True, [{"role": "user", "content": "¿Qué ves?"}], []),
            (False, [{"role": "user", "content": "¿Qué ves?"}], [CAMERA]),
            (True, [{"role": "tool", "content": "¿Qué ves?"}], [CAMERA]),
            (True, [{"role": "user", "content": "¿Puedes usar la cámara?"}], [CAMERA]),
        )
        for enabled, dialogue, tools in scenarios:
            with self.subTest(enabled=enabled, dialogue=dialogue, tools=tools):
                provider = self.make_provider(enabled)
                self.assertEqual(list(provider.response_with_functions(
                    "test-session", dialogue, tools,
                )), [("Respuesta del modelo.", None)])
                provider.client.chat.completions.create.assert_called_once()
