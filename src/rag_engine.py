import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import chromadb
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

# Konfigurasi logging terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag_engine")

# Konstanta path & konfigurasi model
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
NODES_PICKLE_PATH = CHROMA_DB_DIR / "nodes.pkl"
COLLECTION_NAME = "pastoral_katolik"

LLM_MODEL_NAME = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.1
EMBEDDING_CANDIDATE_MODELS = (
    "models/text-embedding-004",
    "gemini-embedding-001",
    "gemini-embedding-2",
)
SIMILARITY_TOP_K = 4
FUSION_NUM_QUERIES = 1

MAX_RETRIES = 5
BASE_RETRY_DELAY = 5.0

NOT_FOUND_MESSAGE = (
    "Maaf, informasi spesifik tersebut belum tersedia dalam dokumen pedoman yang saya miliki."
)

SYSTEM_PROMPT = f"""Anda adalah Asisten Praktis Pastoral & Liturgi Gereja Katolik yang teliti, setia pada ajaran resmi Gereja Katolik, dan hanya berbicara berdasarkan dokumen rujukan yang diberikan.

ATURAN KETAT (WAJIB DIPATUHI):
1. Jawab HANYA berdasarkan potongan dokumen konteks yang disediakan pada pesan pengguna. DILARANG KERAS mengarang, menebak, atau menambahkan informasi dari luar konteks meskipun Anda mengetahuinya.
2. Jika jawaban atas pertanyaan tidak ditemukan secara eksplisit atau tidak cukup didukung oleh konteks yang tersedia, balas PERSIS dengan kalimat berikut tanpa tambahan apa pun:
{NOT_FOUND_MESSAGE}
3. Gunakan bahasa Indonesia yang jelas, sopan, pastoral, dan praktis.
4. Setiap jawaban yang ditemukan di dalam konteks WAJIB disusun mengikuti format terstruktur berikut:
1. Langkah Praktis / Jawaban Utama
2. Hal Penting yang Perlu Diperhatikan
3. Rujukan Dokumen Resmi
5. Pada bagian "3. Rujukan Dokumen Resmi", cantumkan hanya judul dokumen dan nomor halaman yang benar-benar muncul pada konteks yang digunakan."""

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


def is_permission_or_key_error(exc: Exception) -> bool:
    """Mendeteksi error 403 PERMISSION_DENIED atau 400 API_KEY_INVALID."""
    error_text = str(exc).lower()
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return (
        status_code in (400, 403)
        or "permission_denied" in error_text
        or "api_key_invalid" in error_text
        or "403" in error_text
    )


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
                    f"{max_retries} percobaan. Silakan tunggu beberapa saat lalu coba lagi. Detail: {exc}"
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
        raise EnvironmentError(
            "API key Gemini tidak ditemukan atau masih berupa placeholder. "
            "Pastikan GEMINI_API_KEY (atau GOOGLE_API_KEY) sudah diisi di file .env."
        )
    return api_key


def create_embedding_model(api_key: str) -> GoogleGenAIEmbedding:
    """Menginisialisasi GoogleGenAIEmbedding dengan fallback otomatis jika model 404."""
    env_model = os.getenv("EMBEDDING_MODEL", "").strip()
    candidates = [env_model] if env_model else []
    for model_name in EMBEDDING_CANDIDATE_MODELS:
        if model_name not in candidates:
            candidates.append(model_name)

    last_error: Exception | None = None
    for candidate_model in candidates:
        embed_model = GoogleGenAIEmbedding(
            model_name=candidate_model,
            api_key=api_key,
        )
        try:
            retry_with_backoff(
                func=lambda m=embed_model: m.get_text_embedding("uji koneksi embedding"),
                action_label=f"uji model embedding {candidate_model}",
            )
            logger.info("Model embedding aktif: %s", candidate_model)
            return embed_model
        except Exception as exc:
            last_error = exc
            if is_model_not_found_error(exc):
                logger.warning(
                    "Model embedding '%s' mengembalikan 404 pada v1beta, mencoba fallback...",
                    candidate_model,
                )
                continue
            if is_permission_or_key_error(exc):
                raise PermissionError(
                    "Akses Gemini API ditolak (403/400). Periksa API key atau billing project Anda."
                ) from exc
            raise

    raise RuntimeError(
        f"Tidak ada model embedding Gemini yang tersedia ({', '.join(candidates)}). "
        f"Detail terakhir: {last_error}"
    )


def load_nodes_from_pickle(pickle_path: Path) -> list[BaseNode]:
    """Memuat daftar nodes dari ./chroma_db/nodes.pkl untuk BM25Retriever."""
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"File {pickle_path} tidak ditemukan. Jalankan 'python src/ingest.py' terlebih dahulu."
        )

    with pickle_path.open("rb") as file:
        nodes = pickle.load(file)

    if not isinstance(nodes, list) or not nodes:
        raise ValueError(
            f"Isi {pickle_path} kosong atau tidak valid. Jalankan ulang 'python src/ingest.py --reset'."
        )

    logger.info("Berhasil memuat %d nodes dari %s.", len(nodes), pickle_path)
    return nodes


class PastoralRAGEngine:
    """Engine Hybrid RAG (Vector + BM25 + Reciprocal Rerank) untuk Asisten Pastoral Katolik."""

    def __init__(self) -> None:
        logger.info("Menginisialisasi PastoralRAGEngine...")
        api_key = load_api_key()

        if not CHROMA_DB_DIR.exists():
            raise FileNotFoundError(
                f"Direktori ChromaDB tidak ditemukan di {CHROMA_DB_DIR}. "
                "Jalankan 'python src/ingest.py' terlebih dahulu."
            )

        self.embed_model = create_embedding_model(api_key)
        self.llm = GoogleGenAI(
            model=LLM_MODEL_NAME,
            api_key=api_key,
            temperature=LLM_TEMPERATURE,
        )

        # 1. Muat ChromaDB persisten dan Vector Retriever
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        existing_collections = [col.name for col in chroma_client.list_collections()]
        if COLLECTION_NAME not in existing_collections:
            raise ValueError(
                f"Koleksi ChromaDB '{COLLECTION_NAME}' belum ada di {CHROMA_DB_DIR}. "
                "Jalankan 'python src/ingest.py' terlebih dahulu."
            )

        chroma_collection = chroma_client.get_collection(name=COLLECTION_NAME)
        if chroma_collection.count() == 0:
            raise ValueError(
                f"Koleksi ChromaDB '{COLLECTION_NAME}' masih kosong. "
                "Jalankan 'python src/ingest.py --reset' untuk mengisi indeks."
            )

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=self.embed_model,
        )
        vector_retriever = vector_index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)

        # 2. Muat nodes.pkl dan inisialisasi BM25Retriever
        nodes = load_nodes_from_pickle(NODES_PICKLE_PATH)
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=SIMILARITY_TOP_K,
        )

        # 3. Gabungkan dengan QueryFusionRetriever (Hybrid Search: Reciprocal Rerank)
        self.retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            llm=self.llm,
            similarity_top_k=SIMILARITY_TOP_K,
            num_queries=FUSION_NUM_QUERIES,
            mode="reciprocal_rerank",
            use_async=False,
        )
        logger.info(
            "PastoralRAGEngine siap (koleksi='%s', total_vektor=%d, llm='%s').",
            COLLECTION_NAME,
            chroma_collection.count(),
            LLM_MODEL_NAME,
        )

    @staticmethod
    def _build_context_block(retrieved_nodes: list[NodeWithScore]) -> str:
        """Menyusun potongan konteks dokumen beserta metadata untuk prompt LLM."""
        context_parts: list[str] = []
        for idx, item in enumerate(retrieved_nodes, start=1):
            metadata = item.node.metadata or {}
            title = str(metadata.get("title") or metadata.get("file_name") or "Dokumen Tanpa Judul")
            category = str(metadata.get("category") or "")
            page_label = str(metadata.get("page_label") or "1")
            source_url = str(metadata.get("source_url") or "")
            content = item.node.get_content().strip()

            header = (
                f"[Sumber {idx}] Judul: {title} | Halaman: {page_label}"
                + (f" | Kategori: {category}" if category else "")
                + (f" | URL: {source_url}" if source_url else "")
            )
            context_parts.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(context_parts)

    @staticmethod
    def _extract_unique_sources(retrieved_nodes: list[NodeWithScore]) -> list[dict[str, str]]:
        """Mengekstrak daftar sumber unik tanpa duplikat dari node hasil retrieval."""
        sources: list[dict[str, str]] = []
        seen_keys: set[tuple[str, str, str]] = set()

        for item in retrieved_nodes:
            metadata = item.node.metadata or {}
            title = str(metadata.get("title") or metadata.get("file_name") or "Dokumen Tanpa Judul")
            category = str(metadata.get("category") or "")
            page_label = str(metadata.get("page_label") or "1")
            source_url = str(metadata.get("source_url") or "")

            dedup_key = (title, page_label, source_url)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            raw_text = " ".join(item.node.get_content().split())
            snippet = raw_text[:150]

            sources.append(
                {
                    "title": title,
                    "category": category,
                    "page_label": page_label,
                    "source_url": source_url,
                    "snippet": snippet,
                }
            )

        return sources

    def query(self, question: str) -> dict[str, Any]:
        """Menjalankan hybrid retrieval dan generasi jawaban terstruktur dengan Gemini."""
        cleaned_query = question.strip()
        if not cleaned_query:
            return {
                "answer": NOT_FOUND_MESSAGE,
                "sources": [],
                "retrieved_contexts": [],
            }

        retrieved_nodes = retry_with_backoff(
            func=lambda: self.retriever.retrieve(cleaned_query),
            action_label="hybrid retrieval (Vector + BM25)",
        )

        if not retrieved_nodes:
            return {
                "answer": NOT_FOUND_MESSAGE,
                "sources": [],
                "retrieved_contexts": [],
            }

        retrieved_contexts = [item.node.get_content() for item in retrieved_nodes]

        context_block = self._build_context_block(retrieved_nodes)
        user_prompt = (
            f"KONTEKS DOKUMEN RESMI:\n{context_block}\n\n"
            f"PERTANYAAN PENGGUNA:\n{cleaned_query}\n\n"
            "Jawablah pertanyaan pengguna HANYA berdasarkan KONTEKS DOKUMEN RESMI di atas "
            "sesuai instruksi sistem."
        )

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]

        try:
            response = retry_with_backoff(
                func=lambda: self.llm.chat(messages),
                action_label=f"generasi jawaban LLM ({LLM_MODEL_NAME})",
            )
        except Exception as exc:
            if is_permission_or_key_error(exc):
                raise PermissionError(
                    "Akses Gemini API ditolak (403/400) saat memanggil LLM. "
                    "Periksa GEMINI_API_KEY atau billing project Anda."
                ) from exc
            raise

        answer_text = (response.message.content or "").strip()
        if not answer_text:
            answer_text = NOT_FOUND_MESSAGE

        # Jika jawaban menyatakan tidak tersedia di dokumen, kosongkan sources untuk UI,
        # namun tetap kembalikan retrieved_contexts agar RAGAS dapat mengevaluasi retrieval.
        if answer_text == NOT_FOUND_MESSAGE:
            return {
                "answer": answer_text,
                "sources": [],
                "retrieved_contexts": retrieved_contexts,
            }

        sources = self._extract_unique_sources(retrieved_nodes)
        return {
            "answer": answer_text,
            "sources": sources,
            "retrieved_contexts": retrieved_contexts,
        }


_rag_engine_instance: PastoralRAGEngine | None = None


def get_rag_engine() -> PastoralRAGEngine:
    """Mengembalikan instance singleton PastoralRAGEngine agar database cukup dimuat sekali."""
    global _rag_engine_instance
    if _rag_engine_instance is None:
        _rag_engine_instance = PastoralRAGEngine()
    return _rag_engine_instance


def ask_question(query: str) -> dict[str, Any]:
    """Fungsi utama untuk menjawab pertanyaan pastoral/liturgi dengan Hybrid RAG."""
    engine = get_rag_engine()
    return engine.query(query)


if __name__ == "__main__":
    sample_query = (
        "Apa saja syarat dan ketentuan bagi orang sakit berat agar dapat menerima "
        "Sakramen Pengurapan Orang Sakit menurut Kitab Hukum Kanonik atau Katekismus?"
    )
    print("=" * 80)
    print(f"PERTANYAAN UJI:\n{sample_query}")
    print("=" * 80)

    try:
        result = ask_question(sample_query)
    except Exception as error:
        logger.error("Gagal memproses pertanyaan: %s", error)
        sys.exit(1)

    print("\nJAWABAN:\n")
    print(result["answer"])

    print("\n" + "-" * 80)
    print(f"SUMBER RUJUKAN ({len(result['sources'])} dokumen unik):")
    for idx, src in enumerate(result["sources"], start=1):
        print(
            f"\n[{idx}] {src['title']} (Halaman {src['page_label']})"
            + (f" | Kategori: {src['category']}" if src["category"] else "")
        )
        if src["source_url"]:
            print(f"    URL    : {src['source_url']}")
        print(f"    Snippet: {src['snippet']}...")
    print(f"\nRETRIEVED CONTEXTS (RAGAS): {len(result['retrieved_contexts'])} chunk teks penuh")
    print("=" * 80)