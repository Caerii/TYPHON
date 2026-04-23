from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from fla.models.gated_deltanet import GatedDeltaNetForCausalLM


def _load_request(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def _convert_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    conversion: dict[str, Any],
) -> dict[str, torch.Tensor]:
    drop_suffixes = tuple(conversion.get("drop_suffixes", []))
    split_fused_swiglu = bool(conversion.get("split_fused_swiglu_gate_proj", False))
    converted: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if drop_suffixes and key.endswith(drop_suffixes):
            continue
        if (
            split_fused_swiglu
            and key.endswith(".mlp.gate_proj.weight")
            and value.ndim == 2
            and value.shape[0] % 2 == 0
        ):
            half = value.shape[0] // 2
            converted[key] = value[:half].contiguous()
            converted[key.replace(".mlp.gate_proj.weight", ".mlp.up_proj.weight")] = (
                value[half:].contiguous()
            )
            continue
        converted[key] = value
    return converted


def _load_model(payload: dict[str, Any]) -> tuple[GatedDeltaNetForCausalLM, Any, dict[str, Any]]:
    model_config = dict(payload["model"])
    checkpoint_id = str(model_config["checkpoint_id"])
    tokenizer_id = str(model_config.get("tokenizer_id") or checkpoint_id)
    dtype_name = str(model_config.get("dtype", "bfloat16"))
    device_name = str(model_config.get("device", "cuda"))
    trust_remote_code = bool(model_config.get("trust_remote_code", True))
    conversion = dict(model_config.get("checkpoint_conversion", {}))

    config = AutoConfig.from_pretrained(checkpoint_id, trust_remote_code=trust_remote_code)
    if "tie_word_embeddings" in conversion:
        config.tie_word_embeddings = bool(conversion["tie_word_embeddings"])

    model = GatedDeltaNetForCausalLM(config)
    state_path = hf_hub_download(checkpoint_id, "model.safetensors")
    state_dict = load_file(state_path)
    converted_state = _convert_state_dict(
        state_dict,
        conversion=conversion,
    )
    missing, unexpected = model.load_state_dict(converted_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Converted Gated DeltaNet checkpoint did not load cleanly. "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    dtype = _dtype_from_name(dtype_name)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=dtype)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "checkpoint_id": checkpoint_id,
        "tokenizer_id": tokenizer_id,
        "dtype": dtype_name,
    }
    return model, tokenizer, runtime


def _run_generation(
    *,
    model: GatedDeltaNetForCausalLM,
    tokenizer: Any,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    model_config = dict(payload["model"])
    max_input_tokens = int(model_config.get("max_input_tokens", 2048))
    max_output_tokens = int(model_config.get("max_output_tokens", 64))
    temperature = float(model_config.get("temperature", 0.0))
    do_sample = temperature > 0.0
    device = next(model.parameters()).device

    results: list[dict[str, Any]] = []
    for sample in payload["samples"]:
        prompt_text = f"{sample['system_prompt']}\n\n{sample['user_prompt']}"
        try:
            untruncated = tokenizer(prompt_text, return_tensors="pt")
            input_token_count = int(untruncated["input_ids"].shape[1])
            encoded = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            )
            retained_input_tokens = int(encoded["input_ids"].shape[1])
            truncated_prompt = retained_input_tokens < input_token_count
            encoded = {key: value.to(device) for key, value in encoded.items()}

            generate_kwargs: dict[str, Any] = {
                **encoded,
                "max_new_tokens": max_output_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generate_kwargs["temperature"] = temperature

            with torch.no_grad():
                output_ids = model.generate(**generate_kwargs)
            generated_tokens = output_ids[0][encoded["input_ids"].shape[1] :]
            predicted_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            results.append(
                {
                    "sample_id": sample["sample_id"],
                    "predicted_answer": predicted_answer,
                    "usage": {
                        "prompt_tokens": retained_input_tokens,
                        "completion_tokens": int(generated_tokens.shape[0]),
                        "total_tokens": retained_input_tokens + int(generated_tokens.shape[0]),
                    },
                    "input_token_count": input_token_count,
                    "retained_input_tokens": retained_input_tokens,
                    "truncated_prompt": truncated_prompt,
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "sample_id": sample["sample_id"],
                    "predicted_answer": "",
                    "usage": {},
                    "input_token_count": None,
                    "retained_input_tokens": None,
                    "truncated_prompt": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Gated DeltaNet FLA generation job inside WSL.")
    parser.add_argument("--request-file", required=True, help="Input request JSON.")
    parser.add_argument("--response-file", required=True, help="Output response JSON.")
    args = parser.parse_args()

    request_path = Path(args.request_file)
    response_path = Path(args.response_file)
    payload = _load_request(request_path)

    try:
        model, tokenizer, runtime = _load_model(payload)
        samples = _run_generation(
            model=model,
            tokenizer=tokenizer,
            payload=payload,
        )
        response = {
            "status": "ok",
            "runtime": runtime,
            "samples": samples,
        }
    except Exception as exc:  # noqa: BLE001
        response = {
            "status": "error",
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
            },
            "error": f"{type(exc).__name__}: {exc}",
            "samples": [],
        }

    response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
    return 0 if response["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
