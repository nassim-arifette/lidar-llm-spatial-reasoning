"""The same two modes, run against a hosted OpenAI-compatible gateway.

``qwen_runtime`` loads open weights and controls the sampling loop. This is the
alternative: the model runs somewhere else and we send it chat messages. It
exists because the gateway offers checkpoints we cannot host --- a 235B model
does not fit on a 40 GB card at any precision we would accept --- and it
deliberately mirrors :class:`qwen_runtime.QwenRuntime` so that
``compare_llm_modes`` does not know which one it is driving.

What changes, and why it matters for reading the numbers:

**The family changes.** The gateway's catalogue carries no Qwen2.5 at all.
A gateway result is therefore a curve of its own, inside one family, and not
a further point on the open-weights curve: putting a Qwen3-32B next to a
Qwen2.5-1.5B would move family and size together and neither could be
credited.

**Determinism is requested, not guaranteed.** We send ``temperature=0`` and
``top_p=1``, which is the closest a hosted endpoint offers to the greedy
decoding the local runtime performs exactly. Batching, routing and server-side
version changes are outside our control, so a gateway run is reproducible only
to the extent the provider makes it so. The recorded fingerprint says which
backend produced a number for exactly this reason.

**Token counts come from the provider.** The local runtime counts with the
tokenizer it decodes with; here we report ``usage`` as returned. They are not
the same instrument and should not be compared across backends.

The credential is read from the environment and never logged, never written to
a result file, and never passed on a command line. ``.env`` is gitignored; the
repository is public.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .qwen_runtime import Generation

DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh/v1"
DEFAULT_MODEL = "alibaba/qwen-3-32b"
ENV_VAR = "AI_GATEWAY_API_KEY"


def load_env(path: str = ".env") -> List[str]:
    """Read ``KEY=VALUE`` lines into the environment. Returns names only.

    Values are never returned, printed or stored. An exported variable wins
    over the file, because exporting it was a deliberate act.
    """
    if not os.path.exists(path):
        return []
    loaded: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name, value = name.strip(), value.strip().strip('"').strip("'")
            if name and value and name not in os.environ:
                os.environ[name] = value
                loaded.append(name)
    return loaded


@dataclass(frozen=True)
class GatewayConfig:
    """Everything that changes what the endpoint returns."""

    model_name: str = DEFAULT_MODEL
    max_new_tokens: int = 256
    max_input_tokens: int = 32768
    base_url: str = DEFAULT_BASE_URL
    batch_size: int = 8          # concurrent requests, not a batched forward
    timeout: float = 180.0
    retries: int = 3
    decoding: str = "temperature=0"

    def fingerprint(self) -> Dict[str, object]:
        return {"backend": "gateway", "model": self.model_name,
                "max_new_tokens": self.max_new_tokens,
                "max_input_tokens": self.max_input_tokens,
                "base_url": self.base_url, "decoding": self.decoding}

    def identity(self) -> str:
        import hashlib
        blob = json.dumps(self.fingerprint(), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


class GatewayRuntime:
    """A hosted chat model, driven through the same interface as the local one."""

    def __init__(self, config: Optional[GatewayConfig] = None,
                 refuse_overlong: bool = True) -> None:
        self.config = config or GatewayConfig()
        self.refuse_overlong = refuse_overlong
        self.model_revision: Optional[str] = None
        self._key: Optional[str] = None

    # -- credential ------------------------------------------------------- #

    def _authorisation(self) -> str:
        if self._key is None:
            load_env()
            key = os.environ.get(ENV_VAR)
            if not key:
                raise RuntimeError(
                    f"{ENV_VAR} is not set, and the gateway backend needs it. "
                    f"Put it in .env (gitignored) or export it in your shell. "
                    f"Never paste it into a source file or a notebook cell.")
            self._key = key
        return f"Bearer {self._key}"

    # -- interface mirrored from QwenRuntime ------------------------------ #

    def ensure_loaded(self) -> None:
        """Nothing to load; check the credential is there before a long run."""
        self._authorisation()

    @property
    def device(self) -> str:
        return "gateway"

    def count_tokens(self, messages: Sequence[Dict[str, str]]) -> int:
        """A rough character-based estimate.

        The provider's tokenizer is not ours. This is used only to spot a
        prompt that has grown unexpectedly, never reported as a measurement.
        """
        text = "".join(m.get("content", "") for m in messages)
        return len(text) // 4

    # -- generation ------------------------------------------------------- #

    def _one(self, messages: Sequence[Dict[str, str]]) -> Generation:
        payload = json.dumps({
            "model": self.config.model_name,
            "messages": list(messages),
            "max_tokens": self.config.max_new_tokens,
            # The closest a hosted endpoint offers to greedy decoding.
            "temperature": 0,
            "top_p": 1,
        }).encode("utf-8")

        last: Optional[Exception] = None
        for attempt in range(self.config.retries):
            request = urllib.request.Request(
                f"{self.config.base_url}/chat/completions", data=payload,
                headers={"Authorization": self._authorisation(),
                         "Content-Type": "application/json"})
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    body = json.load(response)
                elapsed = time.perf_counter() - start
                choice = body["choices"][0]
                usage = body.get("usage", {}) or {}
                return Generation(
                    text=choice["message"].get("content") or "",
                    input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    output_tokens=int(usage.get("completion_tokens", 0) or 0),
                    truncated_input=False,
                    hit_token_budget=(choice.get("finish_reason") == "length"),
                    seconds=elapsed)
            except urllib.error.HTTPError as exc:
                # The body may carry the provider's explanation; the request
                # headers must never be echoed, they hold the credential.
                detail = exc.read()[:300].decode("utf-8", "replace")
                last = RuntimeError(f"gateway HTTP {exc.code}: {detail}")
                if exc.code in (400, 401, 403, 404):
                    raise last from None
            except Exception as exc:                     # noqa: BLE001
                last = exc
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"gateway failed after {self.config.retries} tries: {last}")

    def generate(
        self, conversations: Sequence[Sequence[Dict[str, str]]]
    ) -> List[Generation]:
        """Send one group of conversations, concurrently, in input order."""
        self.ensure_loaded()
        workers = max(min(self.config.batch_size, len(conversations)), 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._one, conversations))

    def generate_all(
        self, conversations: Sequence[Sequence[Dict[str, str]]],
        verbose: bool = False,
    ) -> List[Generation]:
        out: List[Generation] = []
        size = max(self.config.batch_size, 1)
        for start in range(0, len(conversations), size):
            out.extend(self.generate(conversations[start:start + size]))
            if verbose:
                print(f"  generated {len(out)}/{len(conversations)}", flush=True)
        return out


def list_models(base_url: str = DEFAULT_BASE_URL) -> List[str]:
    """Model ids the gateway offers. Ids only; nothing else is read."""
    runtime = GatewayRuntime(GatewayConfig(base_url=base_url))
    request = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": runtime._authorisation()})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.load(response)
    return sorted(m.get("id", "") for m in body.get("data", []))
