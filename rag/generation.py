"""Local generation via vllm (default) or llama-cpp-python (fallback)."""

from pathlib import Path

from rag.config import Generator as GeneratorCfg
from rag.types import GroundingContext

SYSTEM_PROMPT = (
    "Ты — точный ассистент службы поддержки Альфа-Банка. "
    "Отвечай ТОЛЬКО на основе предоставленного контекста. "
    "Если ответа в контексте нет, ответь дословно: "
    "«В контексте нет данных по этому вопросу.» "
    "Не придумывай факты, не используй внешние знания. "
    "Отвечай по-русски, кратко и по существу."
)


def _user_message(context_str: str, query: str) -> str:
    return f"Контекст:\n{context_str}\n\nВопрос: {query}\n\nОтвет:"


class Generator:
    def __init__(self, cfg: GeneratorCfg, seed: int) -> None:
        self.cfg = cfg
        if cfg.backend == "vllm":
            self._init_vllm(seed)
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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_message(ctx.context_str, ctx.query)},
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, ctx: GroundingContext) -> str:
        return self.generate_batch([ctx])[0]

    def generate_batch(self, contexts: list[GroundingContext]) -> list[str]:
        if self.cfg.backend == "vllm":
            prompts = [self._format_vllm_prompt(ctx) for ctx in contexts]
            outputs = self._llm.generate(prompts, self._sampling_params)
            return [o.outputs[0].text.strip() for o in outputs]
        else:
            return [self._generate_llama_cpp(ctx) for ctx in contexts]

    def _generate_llama_cpp(self, ctx: GroundingContext) -> str:
        if not ctx.context_str.strip():
            return ""
        out = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_message(ctx.context_str, ctx.query)},
            ],
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_new_tokens,
        )
        return out["choices"][0]["message"]["content"].strip()
