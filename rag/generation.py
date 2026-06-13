"""Local generation via vllm, llama-cpp-python, or an OpenAI-compatible HTTP server (sglang)."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag.config import Generator as GeneratorCfg
from rag.types import GroundingContext

# Two scoring levers baked in here. (1) Brevity: references are short (~37 tokens median)
# and the metric zeroes answers >=3x the reference. (2) Abstention calibration: ~32% of
# references are "Нет ответа", and over-answering them (generic advice / hallucinated
# specifics) zeroes the question. The reranker score can't separate answerable from no-data
# (distributions overlap), so the model itself must abstain on vague/complaint/ungrounded
# queries. The no-data phrase MUST match config.length.no_data_phrase (two sources, synced
# by hand). After editing this, re-measure BOTH eval_local buckets — no-data Recall-L should
# rise without the substantive bucket dropping.
SYSTEM_PROMPT = (
    "Ты — точный ассистент службы поддержки Альфа-Банка. "
    "Отвечай ТОЛЬКО на основе предоставленного контекста, по-русски, кратко "
    "(1–3 предложения), только фактами, которые прямо отвечают на вопрос. "
    "Не повторяй вопрос, без вступлений, выводов и воды.\n"
    "Ответь дословно «Нет ответа.» (без пояснений), если выполнено хотя бы одно:\n"
    "— в контексте нет прямого, конкретного ответа на вопрос;\n"
    "— вопрос является жалобой, личной ситуацией или просьбой о помощи без фактического вопроса;\n"
    "— ответ требует данных, которых нет в контексте (статус конкретной операции, суммы, "
    "персональные данные клиента);\n"
    "— ты можешь дать только общий совет («обратитесь в банк», «проверьте настройки»).\n"
    "Не придумывай факты, не давай общих рекомендаций, не используй внешние знания."
)


def _user_message(context_str: str, query: str) -> str:
    return f"Контекст:\n{context_str}\n\nВопрос: {query}\n\nОтвет:"


def _messages(ctx: GroundingContext) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_message(ctx.context_str, ctx.query)},
    ]


class Generator:
    def __init__(self, cfg: GeneratorCfg, seed: int) -> None:
        self.cfg = cfg
        if cfg.backend == "vllm":
            self._init_vllm(seed)
        elif cfg.backend == "openai_api":
            self._init_openai_api()
        else:
            self._init_llama_cpp(seed)

    def _init_vllm(self, seed: int) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model)
        llm_kwargs = dict(
            model=self.cfg.model,
            quantization=self.cfg.quantization or None,
            tensor_parallel_size=self.cfg.tensor_parallel_size,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            enforce_eager=self.cfg.enforce_eager,
            seed=seed,
        )
        if self.cfg.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.cfg.max_model_len
        if self.cfg.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = self.cfg.max_num_seqs
        llm_kwargs["swap_space"] = 4
        self._llm = LLM(**llm_kwargs)
        self._sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_new_tokens,
        )

    def _init_openai_api(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key or "EMPTY",
            timeout=self.cfg.request_timeout,
            max_retries=2,
        )
        self._executor = ThreadPoolExecutor(max_workers=max(1, self.cfg.max_concurrency))

    def _init_llama_cpp(self, seed: int) -> None:
        from llama_cpp import Llama

        model_path = Path(self.cfg.model_path or "")
        if not model_path.is_absolute():
            from rag.config import REPO_ROOT

            model_path = REPO_ROOT / model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"GGUF model not found at {model_path}. "
                "Download it via huggingface-cli before running."
            )
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self.cfg.n_ctx,
            n_gpu_layers=self.cfg.n_gpu_layers,
            seed=seed,
            verbose=False,
        )

    def _format_vllm_prompt(self, ctx: GroundingContext) -> str:
        return self._tokenizer.apply_chat_template(
            _messages(ctx), tokenize=False, add_generation_prompt=True
        )

    def generate(self, ctx: GroundingContext) -> str:
        return self.generate_batch([ctx])[0]

    def generate_batch(self, contexts: list[GroundingContext]) -> list[str]:
        if self.cfg.backend == "vllm":
            prompts = [self._format_vllm_prompt(ctx) for ctx in contexts]
            outputs = self._llm.generate(prompts, self._sampling_params)
            return [o.outputs[0].text.strip() for o in outputs]
        if self.cfg.backend == "openai_api":
            return self._generate_openai_api_batch(contexts)
        return [self._generate_llama_cpp(ctx) for ctx in contexts]

    def _generate_openai_api_batch(self, contexts: list[GroundingContext]) -> list[str]:
        if not contexts:
            return []
        return list(self._executor.map(self._generate_openai_api_one, contexts))

    def _generate_openai_api_one(self, ctx: GroundingContext) -> str:
        if not ctx.context_str.strip():
            return ""
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=_messages(ctx),
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_new_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def _generate_llama_cpp(self, ctx: GroundingContext) -> str:
        if not ctx.context_str.strip():
            return ""
        out = self._llm.create_chat_completion(
            messages=_messages(ctx),
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_new_tokens,
        )
        return out["choices"][0]["message"]["content"].strip()
