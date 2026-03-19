#!/usr/bin/env python3
"""Profile Homura decode performance and produce a summary table.

Usage:
    ./scripts/profile.py Qwen/Qwen3-0.6B
    ./scripts/profile.py Qwen/Qwen3-0.6B --tokens 50 --ctx 4096
    ./scripts/profile.py Qwen/Qwen3-0.6B --raw   # show raw per-kernel lines
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def run_profile(model: str, tokens: int, ctx: int, prompt: str) -> str:
    """Run homura with HOMURA_PROFILE=1 and return stderr."""
    env = os.environ.copy()
    env["HOMURA_PROFILE"] = "1"
    cmd = [
        "cargo",
        "run",
        "--release",
        "--",
        "run",
        model,
        "--prompt",
        prompt,
        "--max-tokens",
        str(tokens),
        "--ctx",
        str(ctx),
        "--temperature",
        "0",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=300
    )
    return result.stderr + result.stdout


def parse_output(output: str) -> dict:
    """Parse profile output into structured data."""
    # Extract all decode kernel profiles
    decode_profiles = []
    current_steps = []
    in_decode = False

    for line in output.splitlines():
        stripped = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()

        if "decode kernel profile" in stripped:
            m = re.search(r"\((\d+) steps, ([\d.]+)ms total\)", stripped)
            if m:
                in_decode = True
                current_steps = []
                total_ms = float(m.group(2))

        elif in_decode and stripped.startswith("│ k"):
            m = re.match(
                r"│ k(\d+)\s+([\d.]+)ms\s+\(\s*([\d.]+)%\)\s+(.*)", stripped
            )
            if m:
                kid = int(m.group(1))
                ms = float(m.group(2))
                pct = float(m.group(3))
                shapes = m.group(4)
                current_steps.append(
                    {"kid": kid, "ms": ms, "pct": pct, "shapes": shapes}
                )

        elif in_decode and "└─" in stripped:
            in_decode = False
            decode_profiles.append(
                {"total_ms": total_ms, "steps": current_steps}
            )

    # Extract summary -- supports both old one-liner and new multi-line format
    summary = {}
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", l).strip() for l in output.splitlines()]
    full_text = " ".join(lines)

    # Old format: "done ... prefill 3.00s ... decode 29 tok in 2.31s (13.2 avg ...)"
    m = re.search(
        r"done.*prefill ([\d.]+)s.*decode (\d+) tok in ([\d.]+)s "
        r"\(([\d.]+) avg.*?([\d.]+) min.*?([\d.]+) max.*?\| ([\d.]+)ms/tok\)",
        full_text,
    )
    if m:
        summary = {
            "prefill_s": float(m.group(1)),
            "decode_tokens": int(m.group(2)),
            "decode_s": float(m.group(3)),
            "avg_tok_s": float(m.group(4)),
            "min_tok_s": float(m.group(5)),
            "max_tok_s": float(m.group(6)),
            "ms_per_tok": float(m.group(7)),
        }
    else:
        # New multi-line format
        prefill_s = decode_tokens = decode_s = avg_tok_s = min_tok_s = max_tok_s = ms_tok = 0.0
        for line in lines:
            m = re.search(r"prefill\s+([\d.]+)s\s+\((\d+) tokens?\)", line)
            if m:
                prefill_s = float(m.group(1))
            m = re.search(r"decode\s+(\d+) tok\s+in ([\d.]+)s", line)
            if m:
                decode_tokens = int(m.group(1))
                decode_s = float(m.group(2))
            m = re.search(r"([\d.]+) avg\s+([\d.]+) min\s+([\d.]+) max", line)
            if m:
                avg_tok_s = float(m.group(1))
                min_tok_s = float(m.group(2))
                max_tok_s = float(m.group(3))
            m = re.search(r"(\d+) ms/tok", line)
            if m:
                ms_tok = float(m.group(1))
        if decode_tokens > 0:
            summary = {
                "prefill_s": prefill_s,
                "decode_tokens": int(decode_tokens),
                "decode_s": decode_s,
                "avg_tok_s": avg_tok_s,
                "min_tok_s": min_tok_s,
                "max_tok_s": max_tok_s,
                "ms_per_tok": ms_tok,
            }

    return {"profiles": decode_profiles, "summary": summary}


def classify_kernel(kid: int, shapes: str, total_kernels: int) -> str:
    """Classify kernel by type based on shapes."""
    if kid == total_kernels - 1:
        # Last compiled kernel is usually LM head
        if "151936" in shapes or "32000" in shapes or "128256" in shapes:
            return "LMHead"
    if kid >= 2**63:
        return "KVConcat"

    # Count weight matrices by shape pattern
    shape_list = shapes.split(" × ")
    num_shapes = len(shape_list)

    if num_shapes >= 8:
        return "QKV"
    elif num_shapes == 5:
        return "Attn"
    elif num_shapes == 6:
        return "MLP"
    elif num_shapes == 3 and any(
        s
        for s in shape_list
        if re.match(r"\[\d+, \d+\]", s.strip()) and "1024" not in s
    ):
        return "LMHead"
    elif num_shapes == 3:
        return "LMHead"

    return "Other"


def get_model_config(model: str) -> dict | None:
    """Try to load model config.json from HF cache."""
    cache = Path.home() / ".cache/huggingface/hub"
    model_dir = cache / f"models--{model.replace('/', '--')}"
    if not model_dir.exists():
        return None
    for config_path in model_dir.rglob("config.json"):
        try:
            with open(config_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
    return None


def compute_bandwidth(config: dict, kernel_type: str, ms: float) -> tuple[float, float]:
    """Return (weight_mb, effective_gb_s) for a kernel type per layer."""
    B = 4  # f32
    h = config.get("hidden_size", 0)
    heads = config.get("num_attention_heads", 0)
    kv_heads = config.get("num_key_value_heads", heads)
    head_dim = config.get("head_dim", h // heads if heads else 0)
    inter = config.get("intermediate_size", 0)
    vocab = config.get("vocab_size", 0)

    if kernel_type == "QKV":
        weight_bytes = (
            h * (heads * head_dim)
            + h * (kv_heads * head_dim)
            + h * (kv_heads * head_dim)
        ) * B
    elif kernel_type == "Attn":
        weight_bytes = (heads * head_dim) * h * B  # O_proj
    elif kernel_type == "MLP":
        weight_bytes = (h * inter * 2 + inter * h) * B
    elif kernel_type == "LMHead":
        weight_bytes = h * vocab * B
    else:
        return 0, 0

    mb = weight_bytes / 1e6
    gb_s = weight_bytes / (ms / 1000) / 1e9 if ms > 0 else 0
    return mb, gb_s


def print_summary(data: dict, model: str, show_raw: bool = False):
    """Print formatted profile summary."""
    profiles = data["profiles"]
    summary = data["summary"]
    config = get_model_config(model)

    if not profiles:
        print("No decode profiles found in output.")
        return

    # Use the last few profiles (warmed up) for analysis
    warm_profiles = profiles[-min(5, len(profiles)) :]
    if not warm_profiles:
        return

    # Find total kernel count from first profile
    max_kid = 0
    for p in profiles:
        for s in p["steps"]:
            if s["kid"] < 2**63:
                max_kid = max(max_kid, s["kid"])
    total_kernels = max_kid + 1

    # Aggregate by kernel type across warm profiles
    type_times = defaultdict(list)
    for p in warm_profiles:
        step_by_type = defaultdict(float)
        for s in p["steps"]:
            ktype = classify_kernel(s["kid"], s["shapes"], total_kernels)
            step_by_type[ktype] += s["ms"]
        for ktype, ms in step_by_type.items():
            type_times[ktype].append(ms)

    # Compute median per type
    type_median = {}
    for ktype, times in type_times.items():
        times.sort()
        type_median[ktype] = times[len(times) // 2]

    layers = config.get("num_hidden_layers", 28) if config else 28
    peak_bw = 77.0  # GB/s DDR5-6000

    # Compute totals
    kernel_total = sum(type_median.values())
    wall = summary.get("ms_per_tok", kernel_total * 1.15)
    overhead = wall - kernel_total

    if HAS_RICH:
        console = Console()

        # Header
        console.print()
        console.print(
            f"  [bold]{model}[/bold] decode profile "
            f"([cyan]{summary.get('avg_tok_s', 0):.1f} tok/s[/cyan], "
            f"[cyan]{wall:.0f}ms/tok[/cyan])"
        )
        console.print()

        # Per-kernel-type table
        table = Table(box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False)
        table.add_column("Kernel", style="bold")
        table.add_column("Time", justify="right")
        table.add_column("%", justify="right")
        table.add_column("Per layer", justify="right")
        table.add_column("Weight", justify="right")
        table.add_column("BW", justify="right")
        table.add_column("% peak", justify="right")

        order = ["MLP", "QKV", "Attn", "LMHead", "KVConcat", "Other"]
        for ktype in order:
            if ktype not in type_median:
                continue
            ms = type_median[ktype]
            pct = ms / wall * 100

            if config and ktype in ("QKV", "Attn", "MLP", "LMHead"):
                per_layer = ms / layers if ktype != "LMHead" else ms
                weight_mb, eff_bw = compute_bandwidth(config, ktype, per_layer)
                bw_pct = eff_bw / peak_bw * 100
                bw_color = "green" if bw_pct > 50 else "yellow" if bw_pct > 25 else "red"
                per_l_str = f"{per_layer:.2f}ms" if ktype != "LMHead" else ""
                table.add_row(
                    f"{ktype} ({layers}L)" if ktype != "LMHead" else "LMHead",
                    f"{ms:.1f}ms",
                    f"{pct:.0f}%",
                    per_l_str,
                    f"{weight_mb:.0f} MB",
                    f"[{bw_color}]{eff_bw:.0f} GB/s[/{bw_color}]",
                    f"[{bw_color}]{bw_pct:.0f}%[/{bw_color}]",
                )
            else:
                table.add_row(
                    ktype,
                    f"{ms:.1f}ms",
                    f"{pct:.0f}%",
                    "",
                    "",
                    "",
                    "",
                )

        # Overhead row
        if overhead > 0:
            table.add_row(
                "Overhead",
                f"{overhead:.1f}ms",
                f"{overhead / wall * 100:.0f}%",
                "",
                "",
                "",
                "",
                style="dim",
            )

        console.print(table)

        # Summary bar
        if config:
            total_weight = sum(
                compute_bandwidth(config, kt, type_median.get(kt, 1))[0]
                for kt in ("QKV", "Attn", "MLP")
            ) * layers
            lm_weight = compute_bandwidth(config, "LMHead", 1)[0]
            total_weight += lm_weight
            theory_ms = total_weight * 1e6 / peak_bw / 1e9 * 1000
            eff_bw = total_weight * 1e6 / (wall / 1000) / 1e9

            console.print()
            console.print(
                f"  Total weights: [bold]{total_weight:.0f} MB[/bold]  "
                f"Theory: [bold]{theory_ms:.0f}ms[/bold] @ {peak_bw:.0f} GB/s  "
                f"Effective: [bold]{eff_bw:.0f} GB/s[/bold] ({eff_bw/peak_bw*100:.0f}%)  "
                f"Gap: [bold]{wall/theory_ms:.1f}x[/bold]"
            )

        # Decode stats
        if summary:
            console.print(
                f"  Prefill: {summary.get('prefill_s', 0):.2f}s  "
                f"Decode: {summary.get('decode_tokens', 0)} tok  "
                f"Range: {summary.get('min_tok_s', 0):.1f}-{summary.get('max_tok_s', 0):.1f} tok/s"
            )
        console.print()

    else:
        # Fallback plain text
        print(f"\n  {model} decode: {wall:.0f}ms/tok, {summary.get('avg_tok_s', 0):.1f} tok/s\n")
        for ktype in ["MLP", "QKV", "Attn", "LMHead", "KVConcat", "Other"]:
            if ktype not in type_median:
                continue
            ms = type_median[ktype]
            print(f"  {ktype:10s}  {ms:6.1f}ms  ({ms/wall*100:4.0f}%)")
        if overhead > 0:
            print(f"  {'Overhead':10s}  {overhead:6.1f}ms  ({overhead/wall*100:4.0f}%)")
        print()

    # Raw per-kernel output
    if show_raw and warm_profiles:
        p = warm_profiles[-1]
        print("  Raw per-kernel (last decode step):")
        for s in p["steps"]:
            if s["ms"] >= 0.3:
                ktype = classify_kernel(s["kid"], s["shapes"], total_kernels)
                print(f"    k{s['kid']:<4d} {s['ms']:6.2f}ms  {ktype:8s}  {s['shapes']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Profile Homura decode performance")
    parser.add_argument("model", help="HuggingFace model ID or local path")
    parser.add_argument("--tokens", type=int, default=30, help="Tokens to generate (default: 30)")
    parser.add_argument("--ctx", type=int, default=2048, help="Context length (default: 2048)")
    parser.add_argument("--prompt", default="What is 2+2?", help="Prompt text")
    parser.add_argument("--raw", action="store_true", help="Show raw per-kernel lines")
    args = parser.parse_args()

    if HAS_RICH:
        console = Console()
        console.print(f"  Profiling [bold]{args.model}[/bold] ({args.tokens} tokens)...", style="dim")
    else:
        print(f"  Profiling {args.model} ({args.tokens} tokens)...")

    output = run_profile(args.model, args.tokens, args.ctx, args.prompt)
    data = parse_output(output)

    if not data["summary"]:
        print("Failed to parse output. Raw stderr:")
        print(output[-2000:])
        sys.exit(1)

    print_summary(data, args.model, show_raw=args.raw)


if __name__ == "__main__":
    main()
