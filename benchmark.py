#!/usr/bin/env python3
"""
LLM Localized — Benchmark Suite

Benchmarks local Ollama models across three capability domains:
  • Coding  — generates code and validates with test cases
  • Text    — summarization, reasoning, instruction following
  • Vision  — image description, OCR (auto-skipped if model lacks vision)

Usage:
    python benchmark.py                          # Run all benchmarks
    python benchmark.py --models llama3.1,mistral  # Specific models only
    python benchmark.py --category coding        # Specific category only
    python benchmark.py --export results.json    # Export results to JSON
"""

import json
import time
import sys
import os
import re
import io
import textwrap
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

try:
    import requests
except ImportError:
    print("Error: 'requests' package is required. Install with:")
    print("  pip install requests")
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "models.json"
TEST_IMAGES_DIR = SCRIPT_DIR / "test_images"

# ── Terminal Colors ──────────────────────────────────────────────────────────

class C:
    """ANSI color codes for terminal output."""
    RED     = "\033[0;31m"
    GREEN   = "\033[0;32m"
    YELLOW  = "\033[1;33m"
    CYAN    = "\033[0;36m"
    MAGENTA = "\033[0;35m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    NC      = "\033[0m"

    @staticmethod
    def disable():
        for attr in ["RED", "GREEN", "YELLOW", "CYAN", "MAGENTA", "BOLD", "DIM", "NC"]:
            setattr(C, attr, "")


# ── Helpers ──────────────────────────────────────────────────────────────────

def header(text: str):
    width = 60
    print(f"\n{C.BOLD}{'━' * width}{C.NC}")
    print(f"{C.BOLD}  {text}{C.NC}")
    print(f"{C.BOLD}{'━' * width}{C.NC}\n")


def sub_header(text: str):
    print(f"\n{C.CYAN}── {text} ──{C.NC}\n")


def info(text: str):
    print(f"  {C.CYAN}ℹ{C.NC}  {text}")


def success(text: str):
    print(f"  {C.GREEN}✔{C.NC}  {text}")


def warn(text: str):
    print(f"  {C.YELLOW}⚠{C.NC}  {text}")


def fail_msg(text: str):
    print(f"  {C.RED}✖{C.NC}  {text}")


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.0f}s"


# ── Config Loading ───────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load model list and settings from models.json."""
    if not CONFIG_PATH.exists():
        print(f"{C.RED}Error:{C.NC} Config file not found: {CONFIG_PATH}")
        print(f"Create it with a list of Ollama model names. See README.md.")
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    return config


# ── Ollama API ───────────────────────────────────────────────────────────────

class OllamaClient:
    """Thin wrapper around the Ollama REST API."""

    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_running(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/", timeout=5)
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    def list_local_models(self) -> list[str]:
        """Return names of locally available models."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def model_info(self, model: str) -> dict:
        """Get model metadata including family info."""
        try:
            r = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model},
                timeout=10,
            )
            r.raise_for_status()
            return r.json()
        except Exception:
            return {}

    def supports_vision(self, model: str) -> bool:
        """Check if a model supports image input (e.g. LLaVA, BakLLaVA)."""
        model_data = self.model_info(model)
        # Check model families for vision/clip capability
        families = model_data.get("details", {}).get("families", [])
        if any(f in families for f in ["clip", "llava"]):
            return True
        # Fallback: check model name heuristics
        name_lower = model.lower()
        vision_keywords = ["llava", "bakllava", "moondream", "cogvlm", "minicpm-v", "gemma4"]
        return any(kw in name_lower for kw in vision_keywords)

    def generate(self, model: str, prompt: str, images: list[str] | None = None,
                 temperature: float = 0.1, num_predict: int = 1024) -> dict:
        """
        Send a generate request. Returns dict with 'response', 'total_duration',
        'eval_count', etc.

        Args:
            images: list of base64-encoded image strings
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        if images:
            payload["images"] = images

        r = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()


# ── Test Image Generation ────────────────────────────────────────────────────

def ensure_test_images() -> dict[str, Path]:
    """
    Create simple test images for vision benchmarks using Pillow.
    Returns a dict mapping test_name -> image_path.
    Falls back gracefully if Pillow isn't available.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return {}

    TEST_IMAGES_DIR.mkdir(exist_ok=True)
    images = {}

    # ── Test image 1: Shapes with labels ─────────────────────────────────
    path_shapes = TEST_IMAGES_DIR / "shapes.png"
    if not path_shapes.exists():
        img = Image.new("RGB", (400, 300), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Red circle
        draw.ellipse([30, 50, 150, 170], fill=(220, 50, 50), outline=(180, 30, 30), width=2)
        draw.text((65, 180), "Circle", fill=(0, 0, 0))
        # Blue rectangle
        draw.rectangle([180, 50, 300, 170], fill=(50, 100, 220), outline=(30, 70, 180), width=2)
        draw.text((210, 180), "Square", fill=(0, 0, 0))
        # Green triangle
        draw.polygon([(350, 170), (310, 50), (390, 50)], fill=(50, 180, 80), outline=(30, 140, 60))
        draw.text((320, 180), "Triangle", fill=(0, 0, 0))
        # Title
        draw.text((120, 10), "Three Shapes", fill=(0, 0, 0))
        img.save(path_shapes)
    images["shapes"] = path_shapes

    # ── Test image 2: Text for OCR ───────────────────────────────────────
    path_text = TEST_IMAGES_DIR / "text_sample.png"
    if not path_text.exists():
        img = Image.new("RGB", (500, 200), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        lines = [
            "The quick brown fox jumps",
            "over the lazy dog.",
            "",
            "Python 3.11 | Ollama | 2025",
        ]
        y = 20
        for line in lines:
            draw.text((30, y), line, fill=(30, 30, 30))
            y += 40
        img.save(path_text)
    images["text_ocr"] = path_text

    # ── Test image 3: Counting objects ───────────────────────────────────
    path_count = TEST_IMAGES_DIR / "counting.png"
    if not path_count.exists():
        img = Image.new("RGB", (400, 300), color=(255, 255, 240))
        draw = ImageDraw.Draw(img)
        draw.text((130, 10), "Count the stars", fill=(0, 0, 0))
        # Draw 5 star-like shapes (simple asterisks as circles)
        star_positions = [(60, 80), (180, 60), (300, 90), (120, 200), (260, 210)]
        for x, y in star_positions:
            draw.ellipse([x - 15, y - 15, x + 15, y + 15], fill=(255, 200, 0),
                         outline=(200, 150, 0), width=2)
        img.save(path_count)
    images["counting"] = path_count

    return images


def image_to_base64(path: Path) -> str:
    """Read an image file and return its base64 encoding."""
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── Code Extraction & Execution ─────────────────────────────────────────────

def extract_code(response: str) -> str:
    """
    Extract Python code from a model response.
    Looks for ```python blocks first, then ``` blocks, then raw code.
    """
    # Try ```python ... ``` blocks
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return "\n".join(matches)

    # Try to find function definitions directly
    lines = response.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            in_code = True
        if in_code:
            code_lines.append(line)
            # Stop if we hit an empty line after dedented code
            if stripped == "" and code_lines and not code_lines[-1].startswith(" "):
                if len(code_lines) > 2:
                    break

    return "\n".join(code_lines) if code_lines else response


def safe_exec(code: str, test_code: str, timeout_hint: float = 5.0) -> tuple[bool, str]:
    """
    Execute generated code + test assertions in a sandboxed namespace.
    Returns (passed: bool, detail: str).
    """
    namespace = {"__builtins__": __builtins__}
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        # Execute the generated code to define functions
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, namespace)

        # Run test assertions
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(test_code, namespace)

        return True, "All assertions passed"

    except AssertionError as e:
        return False, f"Assertion failed: {e}"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TEST DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Coding Tests ─────────────────────────────────────────────────────────────

CODING_TESTS = [
    {
        "id": "code_reverse_string",
        "name": "Reverse a String",
        "prompt": (
            "Write a Python function called `reverse_string` that takes a string "
            "as input and returns the reversed string. Do not use slicing ([::-1]). "
            "Only output the code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            assert reverse_string("hello") == "olleh"
            assert reverse_string("") == ""
            assert reverse_string("a") == "a"
            assert reverse_string("racecar") == "racecar"
            assert reverse_string("Python") == "nohtyP"
        """),
    },
    {
        "id": "code_fibonacci",
        "name": "Fibonacci Number",
        "prompt": (
            "Write a Python function called `fibonacci` that takes an integer n "
            "and returns the nth Fibonacci number (0-indexed, so fibonacci(0)=0, "
            "fibonacci(1)=1, fibonacci(6)=8). Only output the code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            assert fibonacci(0) == 0
            assert fibonacci(1) == 1
            assert fibonacci(6) == 8
            assert fibonacci(10) == 55
            assert fibonacci(15) == 610
        """),
    },
    {
        "id": "code_is_prime",
        "name": "Prime Check",
        "prompt": (
            "Write a Python function called `is_prime` that takes an integer and "
            "returns True if it is a prime number, False otherwise. Handle edge "
            "cases (0, 1, negative numbers). Only output the code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            assert is_prime(2) == True
            assert is_prime(17) == True
            assert is_prime(1) == False
            assert is_prime(0) == False
            assert is_prime(4) == False
            assert is_prime(97) == True
            assert is_prime(-5) == False
        """),
    },
    {
        "id": "code_two_sum",
        "name": "Two Sum",
        "prompt": (
            "Write a Python function called `two_sum` that takes a list of integers "
            "and a target integer. Return a list of two indices whose values sum to "
            "the target. Assume exactly one solution exists. "
            "Only output the code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            result = two_sum([2, 7, 11, 15], 9)
            assert sorted(result) == [0, 1]
            result = two_sum([3, 2, 4], 6)
            assert sorted(result) == [1, 2]
            result = two_sum([3, 3], 6)
            assert sorted(result) == [0, 1]
        """),
    },
    {
        "id": "code_flatten_list",
        "name": "Flatten Nested List",
        "prompt": (
            "Write a Python function called `flatten` that takes a nested list "
            "(which can contain integers and other lists at any depth) and returns "
            "a flat list of all integers. Example: flatten([1, [2, [3, 4], 5]]) "
            "returns [1, 2, 3, 4, 5]. Only output the code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            assert flatten([1, [2, [3, 4], 5]]) == [1, 2, 3, 4, 5]
            assert flatten([]) == []
            assert flatten([1, 2, 3]) == [1, 2, 3]
            assert flatten([[[[1]]]]) == [1]
            assert flatten([1, [2], [3, [4, [5]]]]) == [1, 2, 3, 4, 5]
        """),
    },
]

# ── Text Prompting Tests ─────────────────────────────────────────────────────

TEXT_TESTS = [
    {
        "id": "text_summarize",
        "name": "Summarization",
        "prompt": (
            "Summarize the following passage in exactly one sentence:\n\n"
            "Machine learning is a subset of artificial intelligence that focuses on "
            "building systems that learn from data. Unlike traditional programming where "
            "developers write explicit rules, machine learning algorithms use statistical "
            "methods to find patterns in data. This approach has proven particularly "
            "effective in areas such as image recognition, natural language processing, "
            "and recommendation systems. The field has grown rapidly in recent years "
            "due to increased computing power and the availability of large datasets."
        ),
        "validate": lambda resp: (
            len(resp.split()) <= 60          # concise
            and len(resp.strip()) > 20       # not empty
            and any(kw in resp.lower() for kw in ["machine learning", "ai", "data", "learn"])
        ),
        "criteria": "Concise one-sentence summary mentioning ML/AI and data",
    },
    {
        "id": "text_reasoning",
        "name": "Logical Reasoning",
        "prompt": (
            "Solve this logic puzzle and give only the final answer:\n\n"
            "There are 5 people in a room. Each person shakes hands with every "
            "other person exactly once. How many handshakes occur in total?"
        ),
        "validate": lambda resp: "10" in resp,
        "criteria": "Answer contains '10'",
    },
    {
        "id": "text_instruction",
        "name": "Instruction Following",
        "prompt": (
            "List exactly 3 benefits of exercise. Format each as a numbered list "
            "(1. 2. 3.) with each benefit in one short sentence. Do not include "
            "any introduction or conclusion — just the 3 numbered items."
        ),
        "validate": lambda resp: (
            "1." in resp and "2." in resp and "3." in resp
            and "4." not in resp  # should not have a 4th item
        ),
        "criteria": "Exactly 3 numbered items, no extras",
    },
    {
        "id": "text_factual",
        "name": "Factual Knowledge",
        "prompt": "What is the chemical symbol for gold? Answer in one word only.",
        "validate": lambda resp: "au" in resp.lower().strip(),
        "criteria": "Response contains 'Au'",
    },
    {
        "id": "text_math_word",
        "name": "Math Word Problem",
        "prompt": (
            "A train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. "
            "What is the total distance traveled? Show your work briefly, then state "
            "the final answer as 'Answer: X km'."
        ),
        "validate": lambda resp: "270" in resp,
        "criteria": "Answer contains '270' (150 + 120 = 270 km)",
    },
]

# ── Vision Tests ─────────────────────────────────────────────────────────────

VISION_TESTS = [
    {
        "id": "vision_shapes",
        "image_key": "shapes",
        "name": "Shape Recognition",
        "prompt": "Describe the shapes you see in this image and their colors.",
        "validate": lambda resp: (
            any(s in resp.lower() for s in ["circle", "round"])
            and any(s in resp.lower() for s in ["square", "rectangle"])
        ),
        "criteria": "Identifies at least a circle and a rectangle/square",
    },
    {
        "id": "vision_ocr",
        "image_key": "text_ocr",
        "name": "Text Reading (OCR)",
        "prompt": "Read all the text visible in this image. Output only the text you see.",
        "validate": lambda resp: (
            "fox" in resp.lower() or "quick" in resp.lower() or "brown" in resp.lower()
        ),
        "criteria": "Reads key words like 'fox', 'quick', or 'brown'",
    },
    {
        "id": "vision_counting",
        "image_key": "counting",
        "name": "Object Counting",
        "prompt": "How many yellow circles/dots/stars are in this image? Answer with just the number.",
        "validate": lambda resp: "5" in resp,
        "criteria": "Answers '5'",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# TEST RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

class BenchmarkResult:
    """Stores the result of a single test."""
    def __init__(self, test_id: str, test_name: str, category: str):
        self.test_id = test_id
        self.test_name = test_name
        self.category = category
        self.passed: bool | None = None
        self.skipped: bool = False
        self.skip_reason: str = ""
        self.duration_secs: float = 0.0
        self.detail: str = ""
        self.response: str = ""
        self.eval_tokens: int = 0

    @property
    def status_icon(self) -> str:
        if self.skipped:
            return f"{C.DIM}⊘{C.NC}"
        return f"{C.GREEN}✔{C.NC}" if self.passed else f"{C.RED}✖{C.NC}"

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "category": self.category,
            "passed": self.passed,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "duration_secs": round(self.duration_secs, 2),
            "detail": self.detail,
            "response_preview": self.response[:200] if self.response else "",
            "eval_tokens": self.eval_tokens,
        }


def run_coding_tests(client: OllamaClient, model: str, settings: dict) -> list[BenchmarkResult]:
    """Run all coding benchmark tests against a model."""
    results = []
    sub_header(f"Coding Tests ({len(CODING_TESTS)} tests)")

    for test in CODING_TESTS:
        result = BenchmarkResult(test["id"], test["name"], "coding")
        print(f"  {C.DIM}▸{C.NC} {test['name']}...", end=" ", flush=True)

        try:
            t0 = time.time()
            api_resp = client.generate(
                model=model,
                prompt=test["prompt"],
                temperature=settings.get("temperature", 0.1),
                num_predict=settings.get("num_predict", 1024),
            )
            result.duration_secs = time.time() - t0
            result.response = api_resp.get("response", "")
            result.eval_tokens = api_resp.get("eval_count", 0)

            # Extract and validate code
            code = extract_code(result.response)
            passed, detail = safe_exec(code, test["test_code"])
            result.passed = passed
            result.detail = detail

        except requests.Timeout:
            result.passed = False
            result.detail = "Request timed out"
        except Exception as e:
            result.passed = False
            result.detail = f"Error: {e}"

        # Print inline result
        dur = format_duration(result.duration_secs) if result.duration_secs else ""
        if result.passed:
            print(f"{C.GREEN}PASS{C.NC} {C.DIM}({dur}){C.NC}")
        else:
            print(f"{C.RED}FAIL{C.NC} {C.DIM}({dur}) — {result.detail[:60]}{C.NC}")

        results.append(result)

    return results


def run_text_tests(client: OllamaClient, model: str, settings: dict) -> list[BenchmarkResult]:
    """Run all text prompting benchmark tests against a model."""
    results = []
    sub_header(f"Text Prompting Tests ({len(TEXT_TESTS)} tests)")

    for test in TEXT_TESTS:
        result = BenchmarkResult(test["id"], test["name"], "text")
        print(f"  {C.DIM}▸{C.NC} {test['name']}...", end=" ", flush=True)

        try:
            t0 = time.time()
            api_resp = client.generate(
                model=model,
                prompt=test["prompt"],
                temperature=settings.get("temperature", 0.1),
                num_predict=settings.get("num_predict", 1024),
            )
            result.duration_secs = time.time() - t0
            result.response = api_resp.get("response", "")
            result.eval_tokens = api_resp.get("eval_count", 0)

            # Validate response
            result.passed = test["validate"](result.response)
            result.detail = test["criteria"]

        except requests.Timeout:
            result.passed = False
            result.detail = "Request timed out"
        except Exception as e:
            result.passed = False
            result.detail = f"Error: {e}"

        dur = format_duration(result.duration_secs) if result.duration_secs else ""
        if result.passed:
            print(f"{C.GREEN}PASS{C.NC} {C.DIM}({dur}){C.NC}")
        else:
            print(f"{C.RED}FAIL{C.NC} {C.DIM}({dur}) — {result.detail[:60]}{C.NC}")

        results.append(result)

    return results


def run_vision_tests(client: OllamaClient, model: str, settings: dict,
                     test_images: dict[str, Path]) -> list[BenchmarkResult]:
    """Run vision benchmark tests. Skips entirely if model lacks vision support."""
    results = []

    if not client.supports_vision(model):
        sub_header(f"Vision Tests — {C.DIM}SKIPPED (model does not support vision){C.NC}")
        for test in VISION_TESTS:
            r = BenchmarkResult(test["id"], test["name"], "vision")
            r.skipped = True
            r.skip_reason = "Model does not support vision"
            results.append(r)
        return results

    if not test_images:
        sub_header(f"Vision Tests — {C.DIM}SKIPPED (Pillow not installed, no test images){C.NC}")
        for test in VISION_TESTS:
            r = BenchmarkResult(test["id"], test["name"], "vision")
            r.skipped = True
            r.skip_reason = "Test images unavailable (install Pillow)"
            results.append(r)
        return results

    sub_header(f"Vision Tests ({len(VISION_TESTS)} tests)")

    for test in VISION_TESTS:
        result = BenchmarkResult(test["id"], test["name"], "vision")

        # Check if the required test image exists
        image_path = test_images.get(test["image_key"])
        if not image_path or not image_path.exists():
            result.skipped = True
            result.skip_reason = f"Test image '{test['image_key']}' not found"
            print(f"  {C.DIM}▸{C.NC} {test['name']}... {C.DIM}SKIPPED{C.NC}")
            results.append(result)
            continue

        print(f"  {C.DIM}▸{C.NC} {test['name']}...", end=" ", flush=True)

        try:
            img_b64 = image_to_base64(image_path)
            t0 = time.time()
            api_resp = client.generate(
                model=model,
                prompt=test["prompt"],
                images=[img_b64],
                temperature=settings.get("temperature", 0.1),
                num_predict=settings.get("num_predict", 1024),
            )
            result.duration_secs = time.time() - t0
            result.response = api_resp.get("response", "")
            result.eval_tokens = api_resp.get("eval_count", 0)

            result.passed = test["validate"](result.response)
            result.detail = test["criteria"]

        except requests.Timeout:
            result.passed = False
            result.detail = "Request timed out"
        except Exception as e:
            result.passed = False
            result.detail = f"Error: {e}"

        dur = format_duration(result.duration_secs) if result.duration_secs else ""
        if result.passed:
            print(f"{C.GREEN}PASS{C.NC} {C.DIM}({dur}){C.NC}")
        else:
            print(f"{C.RED}FAIL{C.NC} {C.DIM}({dur}) — {result.detail[:60]}{C.NC}")

        results.append(result)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_model_summary(model: str, results: list[BenchmarkResult]):
    """Print a per-model summary table."""
    categories = {}
    for r in results:
        cat = categories.setdefault(r.category, {"pass": 0, "fail": 0, "skip": 0, "total": 0})
        cat["total"] += 1
        if r.skipped:
            cat["skip"] += 1
        elif r.passed:
            cat["pass"] += 1
        else:
            cat["fail"] += 1

    total_pass = sum(c["pass"] for c in categories.values())
    total_run = sum(c["total"] - c["skip"] for c in categories.values())
    total_time = sum(r.duration_secs for r in results)

    print(f"\n  {C.BOLD}Summary:{C.NC}  ", end="")
    for cat_name, counts in categories.items():
        run = counts["total"] - counts["skip"]
        if run > 0:
            pct = counts["pass"] / run * 100
            color = C.GREEN if pct >= 80 else C.YELLOW if pct >= 50 else C.RED
            print(f"{cat_name}: {color}{counts['pass']}/{run}{C.NC}  ", end="")
        else:
            print(f"{cat_name}: {C.DIM}skipped{C.NC}  ", end="")

    overall_pct = (total_pass / total_run * 100) if total_run > 0 else 0
    color = C.GREEN if overall_pct >= 80 else C.YELLOW if overall_pct >= 50 else C.RED
    print(f"│ {C.BOLD}Overall: {color}{total_pass}/{total_run} ({overall_pct:.0f}%){C.NC}")
    print(f"  {C.DIM}Total time: {format_duration(total_time)}{C.NC}")


def print_final_report(all_results: dict[str, list[BenchmarkResult]], run_start: datetime):
    """Print the comprehensive final benchmark report."""
    header("📊  BENCHMARK RESULTS")

    run_duration = (datetime.now() - run_start).total_seconds()
    print(f"  {C.DIM}Started:  {run_start.strftime('%Y-%m-%d %H:%M:%S')}{C.NC}")
    print(f"  {C.DIM}Duration: {format_duration(run_duration)}{C.NC}")
    print(f"  {C.DIM}Models:   {len(all_results)}{C.NC}")

    # ── Detailed per-test results table ──────────────────────────────────
    categories = ["coding", "text", "vision"]
    model_names = list(all_results.keys())

    # Collect all test names per category
    test_map: dict[str, list[str]] = {}
    for results in all_results.values():
        for r in results:
            tests = test_map.setdefault(r.category, [])
            if r.test_name not in tests:
                tests.append(r.test_name)

    for cat in categories:
        if cat not in test_map:
            continue

        sub_header(f"{cat.upper()} Results")

        # Table header
        name_col_w = max(len(t) for t in test_map[cat]) + 2
        model_col_w = max(max((len(m) for m in model_names), default=10), 10)
        header_line = f"  {'Test':<{name_col_w}}"
        for m in model_names:
            # Truncate long model names
            display_name = m[:model_col_w]
            header_line += f" │ {display_name:^{model_col_w}}"
        print(f"{C.BOLD}{header_line}{C.NC}")
        print(f"  {'─' * name_col_w}" + (f"─┼─{'─' * model_col_w}" * len(model_names)))

        for test_name in test_map[cat]:
            row = f"  {test_name:<{name_col_w}}"
            for m in model_names:
                # Find this test's result for this model
                result = next(
                    (r for r in all_results[m] if r.test_name == test_name),
                    None,
                )
                if result is None:
                    cell = f"{C.DIM}—{C.NC}"
                elif result.skipped:
                    cell = f"{C.DIM}skip{C.NC}"
                elif result.passed:
                    dur = format_duration(result.duration_secs)
                    cell = f"{C.GREEN}✔ {dur}{C.NC}"
                else:
                    dur = format_duration(result.duration_secs)
                    cell = f"{C.RED}✖ {dur}{C.NC}"

                # Pad to column width (accounting for ANSI codes)
                visible_len = len(re.sub(r'\033\[[0-9;]*m', '', cell))
                padding = model_col_w - visible_len
                row += f" │ {cell}{' ' * max(0, padding)}"
            print(row)

    # ── Scoreboard ───────────────────────────────────────────────────────
    sub_header("Scoreboard")

    scoreboard = []
    for model, results in all_results.items():
        total_pass = sum(1 for r in results if r.passed)
        total_run = sum(1 for r in results if not r.skipped)
        total_time = sum(r.duration_secs for r in results)
        total_tokens = sum(r.eval_tokens for r in results)
        pct = (total_pass / total_run * 100) if total_run > 0 else 0
        scoreboard.append((model, total_pass, total_run, pct, total_time, total_tokens))

    # Sort by percentage descending
    scoreboard.sort(key=lambda x: (-x[3], x[4]))

    rank_col = 6
    model_col = max(len(s[0]) for s in scoreboard) + 2
    print(f"  {C.BOLD}{'Rank':<{rank_col}}{'Model':<{model_col}}{'Score':>10}{'Time':>12}{'Tokens':>10}{C.NC}")
    print(f"  {'─' * (rank_col + model_col + 32)}")

    for i, (model, passed, run, pct, total_time, tokens) in enumerate(scoreboard, 1):
        color = C.GREEN if pct >= 80 else C.YELLOW if pct >= 50 else C.RED
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"  {i}.")
        print(
            f"  {medal:<{rank_col}}{model:<{model_col}}"
            f"{color}{passed}/{run} ({pct:.0f}%){C.NC:>10}"
            f"  {format_duration(total_time):>10}"
            f"  {tokens:>8}"
        )

    print()


def export_results(all_results: dict[str, list[BenchmarkResult]], path: str,
                   run_start: datetime):
    """Export benchmark results to a JSON file."""
    export = {
        "timestamp": run_start.isoformat(),
        "duration_secs": (datetime.now() - run_start).total_seconds(),
        "models": {},
    }
    for model, results in all_results.items():
        export["models"][model] = {
            "results": [r.to_dict() for r in results],
            "summary": {
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.skipped and not r.passed),
                "skipped": sum(1 for r in results if r.skipped),
            },
        }

    with open(path, "w") as f:
        json.dump(export, f, indent=2)

    success(f"Results exported to {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark local Ollama models on coding, text, and vision tasks.",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated list of models to test (overrides models.json)",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        choices=["coding", "text", "vision"],
        help="Run only a specific test category",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export results to a JSON file",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable colored output",
    )
    args = parser.parse_args()

    if args.no_color:
        C.disable()

    # ── Banner ───────────────────────────────────────────────────────────
    header("🧪  LLM Localized — Benchmark Suite")

    # ── Load config ──────────────────────────────────────────────────────
    config = load_config()
    settings = config.get("settings", {})
    base_url = settings.get("ollama_base_url", "http://localhost:11434")
    timeout = settings.get("timeout_seconds", 120)

    client = OllamaClient(base_url=base_url, timeout=timeout)

    # ── Check Ollama is running ──────────────────────────────────────────
    if not client.is_running():
        fail_msg(f"Ollama is not running at {base_url}")
        print(f"\n  Start it with: {C.BOLD}ollama serve{C.NC}\n")
        sys.exit(1)
    success(f"Connected to Ollama at {base_url}")

    # ── Resolve model list ───────────────────────────────────────────────
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = config.get("models", [])

    if not model_names:
        fail_msg("No models specified. Add models to models.json or use --models.")
        sys.exit(1)

    # Check which models are available locally
    local_models = client.list_local_models()
    info(f"Found {len(local_models)} model(s) pulled locally")

    available = []
    for model in model_names:
        # Match model name (with or without tag)
        matched = any(
            m == model or m.startswith(f"{model}:") or model.startswith(f"{m.split(':')[0]}")
            for m in local_models
        )
        if matched:
            available.append(model)
            success(f"  {model}")
        else:
            warn(f"  {model} — not pulled, skipping (run: ollama pull {model})")

    if not available:
        fail_msg("No requested models are available locally. Pull them first:")
        for m in model_names:
            print(f"  ollama pull {m}")
        sys.exit(1)

    # ── Prepare test images ──────────────────────────────────────────────
    categories_to_run = (
        [args.category] if args.category else ["coding", "text", "vision"]
    )

    test_images = {}
    if "vision" in categories_to_run:
        info("Generating test images for vision benchmarks...")
        test_images = ensure_test_images()
        if test_images:
            success(f"  {len(test_images)} test image(s) ready")
        else:
            warn("  Pillow not installed — vision tests will be skipped")
            warn("  Install with: pip install Pillow")

    # ── Run benchmarks ───────────────────────────────────────────────────
    run_start = datetime.now()
    all_results: dict[str, list[BenchmarkResult]] = {}

    for model in available:
        header(f"🤖  {model}")

        model_results = []

        if "coding" in categories_to_run:
            model_results.extend(run_coding_tests(client, model, settings))

        if "text" in categories_to_run:
            model_results.extend(run_text_tests(client, model, settings))

        if "vision" in categories_to_run:
            model_results.extend(run_vision_tests(client, model, settings, test_images))

        all_results[model] = model_results
        print_model_summary(model, model_results)

    # ── Final report ─────────────────────────────────────────────────────
    print_final_report(all_results, run_start)

    # ── Export if requested ───────────────────────────────────────────────
    if args.export:
        export_results(all_results, args.export, run_start)


if __name__ == "__main__":
    main()
