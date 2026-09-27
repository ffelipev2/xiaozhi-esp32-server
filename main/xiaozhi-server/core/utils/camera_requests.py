"""Recognize short, explicit Spanish requests to use the device camera."""

import asyncio
import re
import unicodedata


CAMERA_TOOL = "self_camera_take_photo"


async def wait_for_camera_registration(conn, query, timeout=3):
    """Let MCP replies register tools before the first explicit camera request."""
    client = getattr(conn, "mcp_client", None)
    if (
        client is None
        or not getattr(conn.llm, "route_camera_requests", False)
        or not is_camera_request(query)
    ):
        return
    deadline = asyncio.get_running_loop().time() + timeout
    while not await client.is_ready():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0 or conn.client_abort:
            return
        await asyncio.sleep(min(0.05, remaining))

_REQUESTS = (
    r"(?:usa|utiliza|activa) (?:la )?camara",
    r"(?:dime |describe )?(?:que ves|(?:que|quien)(?: es)? lo que ves)"
    r"(?: (?:ahora|con la camara|delante(?: de ti)?|enfrente(?: de ti)?))?",
    r"(?:dime )?que (?:(?:hay|tienes|tengo)"
    r"(?: (?:delante|enfrente)(?: de (?:ti|mi))?| frente a (?:ti|mi))|es esto)",
    r"(?:toma|saca|haz|captura)(?:me)? (?:una )?(?:foto|fotografia)"
    r"(?: (?:con la camara|de esto|de lo que ves))?",
    r"(?:mira|observa|describe) (?:esto|lo que ves|lo que tienes delante|"
    r"lo que tengo enfrente|con la camara)",
    r"lee (?:esto|(?:el )?texto (?:que (?:tienes|tengo) (?:delante|enfrente)|"
    r"(?:de |en )(?:la imagen|la camara)|delante de ti))",
)


def _normalize(query):
    if not isinstance(query, str):
        return ""
    normalized = "".join(
        char for char in unicodedata.normalize("NFD", query.casefold())
        if not unicodedata.combining(char)
    )
    normalized = " ".join(re.sub(r"[^a-z0-9\s]", " ", normalized).split())
    return re.sub(r"^por favor | por favor$", "", normalized)


def is_camera_request(query):
    """Exclude capability questions and negations; other requests go to the LLM."""
    normalized = _normalize(query)
    if re.search(r"\b(?:no|nunca|sin)\b", normalized):
        return False
    return any(re.fullmatch(pattern, normalized) for pattern in _REQUESTS)


def camera_question(query):
    """A capture command becomes an image question after the device takes it."""
    normalized = _normalize(query)
    if is_camera_request(query) and re.match(
        r"^(?:(?:toma|saca|haz|captura)(?:me)?|usa|utiliza|activa)\b", normalized
    ):
        return "Describe brevemente lo que ves en esta imagen."
    return query
