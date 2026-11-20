import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.run_config import RunConfig

# Pastikan root proyek berada di sys.path agar import src.rag_engine valid saat dijalankan langsung
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.rag_engine import ask_question  # noqa: E402

# Konfigurasi logging terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

TEST_DATASET_PATH = BASE_DIR / "data" / "test_dataset.json"
OUTPUT_EXCEL_PATH = BASE_DIR / "hasil_evaluasi_ragas.xlsx"

EVAL_LLM_MODEL = "gemini-2.5-flash"
EVAL_EMBEDDING_CANDIDATES = (
    "models/text-embedding-004",
    "models/gemini-embedding-001",
    "gemini-embedding-001",
)
DELAY_BETWEEN_QUERIES_SECONDS = 6
MAX_RETRIES = 5
BASE_RETRY_DELAY = 5.0

T = TypeVar("T")


def is_rate_limit_error(exc: Exception) -> bool:
    """Mendeteksi apakah exception disebabkan oleh rate limit / kuota API Gemini."""
    error_text = str(exc).lower()
    rate_limit_indicators = (
        "429",
        "rate limit",
        "rate_limit",
        "resource_exhausted",
        "resource exhausted",
        "quota",
        "too many requests",
    )
    if any(indicator in error_text for indicator in rate_limit_indicators):
        return True

    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return status_code == 429


def is_model_not_found_error(exc: Exception) -> bool:
    """Mendeteksi error 404 model tidak ditemukan / tidak didukung di API."""
    error_text = str(exc).lower()
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return status_code == 404 or "404" in error_text or "not_found" in error_text


def retry_with_backoff(
    func: Callable[[], T],
    action_label: str,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_RETRY_DELAY,
) -> T:
    """Menjalankan fungsi dengan exponential backoff saat terkena API rate limit."""
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise

            if attempt == max_retries:
                raise RuntimeError(
                    f"Rate limit Gemini API terlampaui saat '{action_label}' setelah "
                    f"{max_retries} percobaan. Detail: {exc}"
                ) from exc

            jitter = random.uniform(0.5, 2.0)
            wait_seconds = base_delay * (2 ** (attempt - 1)) + jitter
            logger.warning(
                "[RATE LIMIT] Terkena batas request API saat '%s' (percobaan %d/%d). "
                "Menunggu %.1f detik sebelum mencoba ulang...",
                action_label,
                attempt,
                max_retries,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Gagal menyelesaikan '{action_label}' setelah {max_retries} percobaan.")


def load_api_key() -> str:
    """Memuat GEMINI_API_KEY atau GOOGLE_API_KEY dari file .env."""
    load_dotenv(dotenv_path=BASE_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key in {"MY_GEMINI_API_KEY", "your_api_key_here"}:
        logger.error(
            "API key Gemini tidak ditemukan atau masih berupa placeholder. "
            "Pastikan GEMINI_API_KEY (atau GOOGLE_API_KEY) sudah diisi di file .env."
        )
        sys.exit(1)

    os.environ["GOOGLE_API_KEY"] = api_key
    return api_key


def load_test_dataset(dataset_path: Path) -> list[dict[str, str]]:
    """Membaca dan memvalidasi data/test_dataset.json."""
    if not dataset_path.exists():
        logger.error("File dataset uji tidak ditemukan di: %s", dataset_path)
        sys.exit(1)

    try:
        with dataset_path.open("r", encoding="utf-8") as file:
            raw_items = json.load(file)
    except json.JSONDecodeError as exc:
        logger.error("Format JSON pada %s tidak valid: %s", dataset_path, exc)
        sys.exit(1)

    if not isinstance(raw_items, list) or not raw_items:
        logger.error("Isi %s harus berupa JSON array yang tidak kosong.", dataset_path)
        sys.exit(1)

    validated_items: list[dict[str, str]] = []
    for idx, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        ground_truth = str(item.get("ground_truth") or "").strip()
        if not question or not ground_truth:
            logger.warning("Entri ke-%d dilewati karena field question/ground_truth kosong.", idx)
            continue
        validated_items.append({"question": question, "ground_truth": ground_truth})

    if not validated_items:
        logger.error("Tidak ada pasangan question dan ground_truth yang valid di %s.", dataset_path)
        sys.exit(1)

    logger.info("Berhasil memuat %d pertanyaan uji dari %s.", len(validated_items), dataset_path)
    return validated_items


def create_ragas_evaluators(
    api_key: str,
) -> tuple[Any, Any]:
    """Menyiapkan evaluator LLM & Embeddings Gemini yang dibungkus wrapper RAGAS."""
    llm = ChatGoogleGenerativeAI(
        model=EVAL_LLM_MODEL,
        google_api_key=api_key,
        temperature=0.0,
    )
    evaluator_llm = LangchainLLMWrapper(llm)

    env_embed_model = os.getenv("EMBEDDING_MODEL", "").strip()
    candidates = [env_embed_model] if env_embed_model else []
    for model_name in EVAL_EMBEDDING_CANDIDATES:
        if model_name not in candidates:
            candidates.append(model_name)

    last_error: Exception | None = None
    for candidate_model in candidates:
        logger.info("Menguji embedding evaluator RAGAS: %s...", candidate_model)
        embeddings = GoogleGenerativeAIEmbeddings(
            model=candidate_model,
            google_api_key=api_key,
        )
        try:
            retry_with_backoff(
                func=lambda e=embeddings: e.embed_query("uji koneksi evaluator ragas"),
                action_label=f"uji embedding evaluator {candidate_model}",
            )
            logger.info("Embedding evaluator aktif: %s", candidate_model)
            return evaluator_llm, LangchainEmbeddingsWrapper(embeddings)
        except Exception as exc:
            last_error = exc
            if is_model_not_found_error(exc):
                logger.warning(
                    "Model embedding evaluator '%s' mengembalikan 404 pada API, "
                    "mencoba fallback berikutnya...",
                    candidate_model,
                )
                continue
            raise

    logger.error(
        "Gagal menginisialisasi model embedding evaluator (%s). Detail: %s",
        ", ".join(candidates),
        last_error,
    )
    sys.exit(1)


def collect_rag_responses(test_items: list[dict[str, str]]) -> Dataset:
    """Menjalankan ask_question() untuk tiap pertanyaan uji dengan jeda anti-rate-limit."""
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    total = len(test_items)
    for idx, item in enumerate(test_items, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info("[%d/%d] Menjalankan RAG untuk pertanyaan: %s", idx, total, question)
        rag_result = retry_with_backoff(
            func=lambda q=question: ask_question(q),
            action_label=f"ask_question ke-{idx}/{total}",
        )

        answer = str(rag_result.get("answer") or "").strip()
        retrieved_contexts = [
            str(ctx) for ctx in (rag_result.get("retrieved_contexts") or []) if str(ctx).strip()
        ]
        if not retrieved_contexts:
            retrieved_contexts = [""]

        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(ground_truth)

        if idx < total:
            logger.info(
                "Menunggu %d detik sebelum pertanyaan berikutnya (menjaga rate limit Free Tier)...",
                DELAY_BETWEEN_QUERIES_SECONDS,
            )
            time.sleep(DELAY_BETWEEN_QUERIES_SECONDS)

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def save_results_to_excel(df: pd.DataFrame, output_path: Path) -> None:
    """Menyimpan DataFrame rincian skor RAGAS ke file Excel (.xlsx)."""
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        logger.error(
            "Library 'openpyxl' belum terpasang untuk menyimpan file .xlsx. "
            "Jalankan: pip install openpyxl"
        )
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    logger.info("Rincian hasil evaluasi berhasil disimpan ke %s.", output_path)


def print_summary(df: pd.DataFrame) -> None:
    """Mencetak rata-rata skor keseluruhan dan rincian per metrik di terminal."""
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision"]

    print("\n" + "=" * 80)
    print("HASIL EVALUASI KUANTITATIF RAGAS (GEMINI 2.5 FLASH EVALUATOR)")
    print("=" * 80)

    averages: dict[str, float] = {}
    for col in metric_cols:
        if col in df.columns:
            mean_val = pd.to_numeric(df[col], errors="coerce").mean(skipna=True)
            averages[col] = float(mean_val) if pd.notna(mean_val) else 0.0
            print(f"  - {col:<20}: {averages[col]:.4f}")

    if averages:
        overall_mean = sum(averages.values()) / len(averages)
        print("-" * 80)
        print(f"  Rata-rata Keseluruhan : {overall_mean:.4f}")

    print("=" * 80 + "\n")


def main() -> None:
    start_time = time.time()
    logger.info("=== Memulai Evaluasi Kuantitatif RAGAS ===")

    api_key = load_api_key()
    test_items = load_test_dataset(TEST_DATASET_PATH)

    dataset = collect_rag_responses(test_items)
    evaluator_llm, evaluator_embeddings = create_ragas_evaluators(api_key)

    run_config = RunConfig(
        max_workers=1,
        timeout=120,
        max_retries=5,
        max_wait=60,
    )

    logger.info(
        "Menjalankan RAGAS evaluate() secara berurutan (max_workers=1, timeout=120) "
        "untuk metrik: faithfulness, answer_relevancy, context_precision..."
    )

    try:
        eval_result: Any = retry_with_backoff(
            func=lambda: evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                run_config=run_config,
            ),
            action_label="RAGAS evaluate()",
        )
    except Exception as exc:
        if is_rate_limit_error(exc):
            logger.error(
                "[RATE LIMIT] Evaluasi RAGAS gagal karena batas kuota Free Tier Gemini API tercapai: %s",
                exc,
            )
        else:
            logger.error("Gagal menjalankan evaluasi RAGAS: %s", exc)
        sys.exit(1)

    results_df = eval_result.to_pandas()
    print_summary(results_df)
    save_results_to_excel(results_df, OUTPUT_EXCEL_PATH)

    elapsed = time.time() - start_time
    logger.info("=== Evaluasi RAGAS Selesai dalam %.2f detik ===", elapsed)


if __name__ == "__main__":
    main()