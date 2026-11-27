"""One way to load and run the language model, shared by both modes.

The comparison is between two *uses* of a model, so everything about the model
has to be the same on both sides: the checkpoint, the tokenizer, the dtype, the
decoding. If the direct mode were loaded in float32 and the program mode in
float16, or one sampled and the other did not, the table would be measuring
that instead.

Two executors that had to agree is how the status filter was lost once in this
project. The same risk applies here, so there is one loader and both modes call
it.

What this module reports, and why each one is reported rather than handled
quietly:

``input_tokens`` / ``truncated_input``
    The direct mode's prompt carries the whole object list and can approach the
    context window. Truncating it silently would delete objects from the scene
    the model is being asked about, and the loss would appear as a reasoning
    error. An over-long prompt is refused by default.
``output_tokens`` / ``hit_token_budget``
    A generation that stops because it ran out of budget is cut off
    mid-structure. Measured on an earlier run of the program mode, a budget of
    96 tokens produced 26% unparseable output, 97% of it truncated JSON -- a
    number that looked like a model failure and was a configuration choice.
    Every generation now says whether it ended on its own.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

#: Frozen for the first comparison. Both modes use this checkpoint; a bigger
#: model is a later experiment and changing it here changes both sides at once,
#: which is the point -- model size and architecture must not move together.
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class RuntimeConfig:
    """Everything that changes what the model emits."""

    model_name: str = DEFAULT_MODEL
    max_new_tokens: int = 256
    #: Prompts longer than this are refused, not trimmed. Qwen2.5 carries a
    #: 32k window; the limit here is about keeping the two modes' inputs
    #: comparable and catching a scene that grew unexpectedly, not about the
    #: model's capacity.
    max_input_tokens: int = 8192
    device: Optional[str] = None
    batch_size: int = 8
    #: Greedy, always. Sampling would make the run unreproducible and a
    #: best-of-n search would give one mode more attempts than the other.
    decoding: str = "greedy"
    #: Weight precision, recorded because it is a property of the run and not
    #: of the model. A 32B checkpoint does not fit on one 40 GB card in
    #: bfloat16, so a size comparison that quantises only the large model
    #: changes two things at once; the intermediate 7B point exists to keep
    #: precision fixed across at least one size step.
    precision: str = "fp16"          # fp16 | bf16 | awq | gptq | int8 | int4
    #: Shard across devices and offload instead of calling ``.to(device)``.
    #: Required for anything that does not fit in one allocation.
    device_map: Optional[str] = None
    #: For chat templates with a thinking switch (Qwen3). ``None`` leaves the
    #: template's default and is omitted from the fingerprint, so runs on
    #: models without the switch keep the identity they always had.
    enable_thinking: Optional[bool] = None

    def fingerprint(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "model": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "max_input_tokens": self.max_input_tokens,
            "precision": self.precision,
            "decoding": self.decoding}
        if self.enable_thinking is not None:
            out["enable_thinking"] = self.enable_thinking
        return out

    def identity(self) -> str:
        blob = json.dumps(self.fingerprint(), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


@dataclass
class Generation:
    """One model output, with the facts needed to trust or distrust it."""

    text: str
    #: Content of a ``<think>`` block, when the model emitted one. It is kept
    #: out of ``text`` so no reader mistakes working for the answer, and its
    #: tokens are still counted in ``output_tokens``.
    thinking: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    truncated_input: bool = False
    hit_token_budget: bool = False
    seconds: float = 0.0

    @property
    def complete(self) -> bool:
        """Whether the model stopped on its own, within an intact prompt."""
        return not self.truncated_input and not self.hit_token_budget

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def split_thinking(text: str):
    """Separate a ``<think>...</think>`` block from the answer that follows.

    An unterminated block (the budget ran out while thinking) leaves no answer
    at all, which is what happened and is what gets scored.
    """
    if "</think>" in text:
        thinking, _, rest = text.rpartition("</think>")
        return rest.strip(), thinking.replace("<think>", "").strip()
    if "<think>" in text:
        return "", text.replace("<think>", "").strip()
    return text, ""


class QwenRuntime:
    """A frozen chat model, loaded once, run greedily, reporting its limits."""

    def __init__(self, config: Optional[RuntimeConfig] = None,
                 refuse_overlong: bool = True) -> None:
        self.config = config or RuntimeConfig()
        # An over-long prompt is an error by default. Allowing it to be trimmed
        # has to be an explicit choice, because the trimming deletes objects.
        self.refuse_overlong = refuse_overlong
        self._tokenizer = None
        self._model = None
        self._device = self.config.device
        self.model_revision: Optional[str] = None

    # -- loading ---------------------------------------------------------- #

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"    # required for batched generation

        kwargs: Dict[str, object] = {}
        precision = self.config.precision.lower()
        if precision in ("int4", "int8"):
            # Quantise on load. A 32B checkpoint is 65 GB in bfloat16 and does
            # not fit on one 40 GB card, so the large point of a size curve
            # cannot be measured at the small point's precision. That is a
            # confound, and it is why the curve has an intermediate model
            # measured in fp16 like the small one.
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=(precision == "int4"),
                load_in_8bit=(precision == "int8"),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = self.config.device_map or "auto"
            dtype = None
        elif precision in ("awq", "gptq"):
            # Pre-quantised checkpoints carry their own config; passing a dtype
            # would override it. They are also too large to materialise and
            # then move, so they are placed as they load.
            kwargs["device_map"] = self.config.device_map or "auto"
            dtype = None
        else:
            dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}.get(
                precision, torch.float16 if self._device == "cuda"
                else torch.float32)
            if self.config.device_map:
                kwargs["device_map"] = self.config.device_map

        # transformers renamed `torch_dtype` to `dtype` in v5; support both so
        # the run is not tied to one release.
        def _load(dtype_key: Optional[str]):
            extra = dict(kwargs)
            if dtype is not None and dtype_key is not None:
                extra[dtype_key] = dtype
            return AutoModelForCausalLM.from_pretrained(
                self.config.model_name, **extra)

        try:
            model = _load("dtype")
        except TypeError:
            model = _load("torch_dtype")

        if "device_map" not in kwargs:
            model = model.to(self._device)
        self._model = model.eval()
        self.model_revision = getattr(
            getattr(self._model, "config", None), "_commit_hash", None)

    @property
    def device(self) -> str:
        self.ensure_loaded()
        return str(self._device)

    def _template(self, messages: Sequence[Dict[str, str]]) -> str:
        extra = ({} if self.config.enable_thinking is None
                 else {"enable_thinking": self.config.enable_thinking})
        return self._tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, **extra)

    def count_tokens(self, messages: Sequence[Dict[str, str]]) -> int:
        self.ensure_loaded()
        return len(self._tokenizer(self._template(messages))["input_ids"])

    # -- generation ------------------------------------------------------- #

    def generate(
        self, conversations: Sequence[Sequence[Dict[str, str]]],
        max_new_tokens: Optional[int] = None,
        prefix_allowed_tokens_fn=None,
    ) -> List[Generation]:
        """Run one batch of chat conversations and return their completions.

        ``max_new_tokens`` overrides the configured budget for this call; the
        caller is then responsible for recording it. ``prefix_allowed_tokens_fn``
        constrains decoding (see ``constrained.py``) and is likewise the
        caller's to record: a constrained run is a separate condition.
        """
        import torch

        self.ensure_loaded()
        budget = max_new_tokens or self.config.max_new_tokens
        prompts = [self._template(messages) for messages in conversations]

        lengths = [len(self._tokenizer(p)["input_ids"]) for p in prompts]
        overlong = [i for i, n in enumerate(lengths)
                    if n > self.config.max_input_tokens]
        if overlong and self.refuse_overlong:
            raise ValueError(
                f"{len(overlong)} prompt(s) exceed max_input_tokens="
                f"{self.config.max_input_tokens} (longest {max(lengths)}). "
                f"Refusing rather than truncating: trimming the prompt deletes "
                f"objects from the scene and the loss would be scored as a "
                f"reasoning error. Raise max_input_tokens deliberately, or "
                f"shorten the scene description for BOTH modes.")

        start = time.perf_counter()
        # max_length is only meaningful alongside truncation; passing it when
        # we refuse to truncate produces a warning and no effect.
        tokenizer_kwargs = ({"truncation": True,
                             "max_length": self.config.max_input_tokens}
                            if not self.refuse_overlong else {"truncation": False})
        # With device_map the embedding layer decides where inputs must live,
        # so ask the model rather than assuming the configured device.
        target = getattr(self._model, "device", self._device)
        batch = self._tokenizer(prompts, return_tensors="pt", padding=True,
                                **tokenizer_kwargs).to(target)
        with torch.no_grad():
            out = self._model.generate(
                **batch,
                max_new_tokens=budget,
                do_sample=False,               # greedy: the run is reproducible
                pad_token_id=self._tokenizer.pad_token_id,
                **({"prefix_allowed_tokens_fn": prefix_allowed_tokens_fn}
                   if prefix_allowed_tokens_fn is not None else {}),
            )
        elapsed = time.perf_counter() - start

        prompt_length = batch["input_ids"].shape[1]
        generated = out[:, prompt_length:]
        texts = self._tokenizer.batch_decode(generated, skip_special_tokens=True)

        # transformers 5 exposes `eos_token_ids` as a plain int on some
        # tokenizers and a list on others; both have to be accepted, or the
        # stop detection raises and takes the whole run with it.
        eos_ids = {self._tokenizer.eos_token_id}
        extra = getattr(self._tokenizer, "eos_token_ids", None)
        if isinstance(extra, int):
            eos_ids.add(extra)
        elif extra:
            eos_ids.update(extra)
        eos_ids.discard(None)

        results: List[Generation] = []
        per_item = elapsed / max(len(texts), 1)
        for i, text in enumerate(texts):
            row = generated[i].tolist()
            produced = [t for t in row if t != self._tokenizer.pad_token_id]
            stopped_on_eos = any(t in eos_ids for t in row)
            text, thinking = split_thinking(text)
            results.append(Generation(
                text=text,
                thinking=thinking,
                input_tokens=lengths[i],
                output_tokens=len(produced),
                truncated_input=lengths[i] > self.config.max_input_tokens,
                # Nothing was emitted to end the sequence and the budget is
                # spent: the output is cut off, whatever it looks like.
                hit_token_budget=(not stopped_on_eos
                                  and len(produced) >= budget),
                seconds=per_item,
            ))
        return results

    def generate_all(
        self, conversations: Sequence[Sequence[Dict[str, str]]],
        verbose: bool = False,
        max_new_tokens: Optional[int] = None,
        prefix_allowed_tokens_fn=None,
    ) -> List[Generation]:
        """Batched :meth:`generate` over an arbitrary number of conversations."""
        out: List[Generation] = []
        size = max(self.config.batch_size, 1)
        for start in range(0, len(conversations), size):
            out.extend(self.generate(
                conversations[start:start + size], max_new_tokens=max_new_tokens,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn))
            if verbose:
                print(f"  generated {len(out)}/{len(conversations)}", flush=True)
        return out
