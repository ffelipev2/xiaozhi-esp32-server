"""Check the camera request's language and image payload without running a model."""

import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.api.vision_handler import VisionHandler
from core.providers.vllm.openai import VLLMProvider


class VisionLanguageTests(unittest.TestCase):
    def make_provider(self, **options):
        with patch("core.providers.vllm.openai.openai.OpenAI") as factory:
            provider = VLLMProvider({
                "model_name": "local-vision",
                "api_key": "ollama",
                "base_url": "http://localhost:11434/v1",
                **options,
            })
            factory.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Una taza."))]
            )
        return provider

    def test_spanish_prompt_preserves_question_and_image(self):
        provider = self.make_provider(system_prompt="Responde en español.")
        self.assertEqual(provider.response("¿Qué ves?", "jpeg-data"), "Una taza.")
        request = provider.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["messages"][0], {
            "role": "system", "content": "Responde en español."
        })
        parts = request["messages"][1]["content"]
        self.assertEqual(parts[0]["text"], "¿Qué ves?")
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/jpeg;base64,jpeg-data")

    def test_default_language_remains_available(self):
        provider = self.make_provider()
        provider.response("Question", "image")
        messages = provider.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], "请使用中文回复。")

    def test_empty_system_prompt_adds_no_language_instruction(self):
        provider = self.make_provider(system_prompt="")
        provider.response("Question", "image")
        messages = provider.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")

    def test_generation_limits_are_sent_to_model(self):
        provider = self.make_provider(max_tokens="300", temperature="0.2", top_p="0.8")
        provider.response("Question", "image")
        request = provider.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["max_tokens"], 300)
        self.assertEqual(request["temperature"], 0.2)
        self.assertEqual(request["top_p"], 0.8)


class VisionResponsivenessTests(unittest.IsolatedAsyncioTestCase):
    async def test_inference_runs_outside_server_event_loop(self):
        loop_thread = threading.get_ident()
        worker_threads = []

        def infer(question, image):
            worker_threads.append(threading.get_ident())
            return "Una taza."

        handler = VisionHandler({
            "server": {"auth_key": "test-key"},
            "selected_module": {"VLLM": "LocalVision"},
            "VLLM": {"LocalVision": {"type": "openai"}},
        })
        reader = SimpleNamespace(next=AsyncMock(side_effect=[
            SimpleNamespace(text=AsyncMock(return_value="¿Qué ves?")),
            SimpleNamespace(read=AsyncMock(return_value=b"jpeg-data")),
        ]))
        request = SimpleNamespace(
            headers={"Device-Id": "test-device", "Client-Id": "test-client"},
            multipart=AsyncMock(return_value=reader),
        )
        with patch.object(handler, "_verify_auth_token", return_value=(True, "test-device")), \
             patch("core.api.vision_handler.is_valid_image_file", return_value=True), \
             patch("core.api.vision_handler.create_instance", return_value=SimpleNamespace(response=infer)):
            response = await handler.handle_post(request)
        self.assertEqual(json.loads(response.text)["response"], "Una taza.")
        self.assertEqual(len(worker_threads), 1)
        self.assertNotEqual(worker_threads[0], loop_thread)


if __name__ == "__main__":
    unittest.main()
