import argparse
import hashlib
import json
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Callable, TypeVar

import chromadb
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Konfigurasi logging terminal agar progres terbaca jelas
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")

# Konstanta path dan konfigurasi RAG
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
METADATA_FILE = BASE_DIR / "data" / "metadata.json"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
NODES_PICKLE_PATH = CHROMA_DB_DIR / "nodes.pkl"
COLLECTION_NAME = "pastoral_katolik"

# Model embedding prioritas: mencoba text-embedding-004 lalu fallback otomatis jika 404 (deprecated di v1beta)
EMBEDDING_CANDIDATE_MODELS = (
    "models/text-embedding-004",
    "gemini-embedding-001",
    "gemini-embedding-2",
)
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
BATCH_SIZE = 50
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
                logger.error(
                    "[RATE LIMIT] Gagal pada tahap '%s' setelah %d kali percobaan. "
                    "Kuota/rate limit Google GenAI API terlampaui. "
                    "Silakan tunggu beberapa menit atau periksa kuota API key Anda. Detail: %s",
                    action_label,
                    max_retries,
                    exc,
                )
                raise RuntimeError(
                    f"Rate limit API terlampaui saat {action_label}: {exc}"
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
    return api_key


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


def create_embedding_model(api_key: str) -> tuple[GoogleGenAIEmbedding, str]:
    """Menginisialisasi GoogleGenAIEmbedding dan memvalidasi model aktif di Gemini API."""
    env_model = os.getenv("EMBEDDING_MODEL", "").strip()
    candidates = [env_model] if env_model else []
    for model_name in EMBEDDING_CANDIDATE_MODELS:
        if model_name not in candidates:
            candidates.append(model_name)

    last_error: Exception | None = None
    for candidate_model in candidates:
        logger.info("Menguji model embedding: %s...", candidate_model)
        embed_model = GoogleGenAIEmbedding(
            model_name=candidate_model,
            api_key=api_key,
        )
        try:
            retry_with_backoff(
                func=lambda m=embed_model: m.get_text_embedding("uji koneksi embedding"),
                action_label=f"uji model embedding {candidate_model}",
            )
            logger.info("Model embedding aktif terpilih: %s", candidate_model)
            return embed_model, candidate_model
        except Exception as exc:
            last_error = exc
            if is_model_not_found_error(exc):
                logger.warning(
                    "Model '%s' mengembalikan 404 (tidak tersedia/deprecated pada v1beta). "
                    "Mencoba model fallback berikutnya...",
                    candidate_model,
                )
                continue
            if is_permission_or_key_error(exc):
                logger.error(
                    "[AKSES DITOLAK / API KEY] Project Google Cloud dari API key Anda tidak memiliki akses "
                    "ke Gemini API (403/400). Jika pesan menyebut 'free tier is not available for this project', "
                    "buat API key di project baru via https://aistudio.google.com/apikey (akun Gmail pribadi) "
                    "atau aktifkan billing di https://aistudio.google.com/billing.\nDetail: %s",
                    exc,
                )
                sys.exit(1)
            raise

    logger.error(
        "Tidak ada model embedding Gemini yang berhasil dihubungi (%s). Detail terakhir: %s",
        ", ".join(candidates),
        last_error,
    )
    sys.exit(1)


def load_metadata_map(metadata_path: Path) -> dict[str, dict[str, str]]:
    """Membaca data/metadata.json dan memetakannya berdasarkan nama file.

    Mendukung format:
    1. JSON array: [{"filename": "...", "title": "...", ...}, ...]
    2. JSON object per filename: {"file.pdf": {"title": "...", ...}, ...}
    3. JSON wrapper object: {"documents": [...]} / {"metadata": [...]} / dll.
    """
    if not metadata_path.exists():
        logger.warning(
            "File metadata tidak ditemukan di %s. Semua dokumen akan memakai metadata default.",
            metadata_path,
        )
        return {}

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except json.JSONDecodeError as exc:
        logger.error("Format JSON pada %s tidak valid: %s", metadata_path, exc)
        sys.exit(1)

    metadata_map: dict[str, dict[str, str]] = {}

    def extract_filename(entry_obj: dict[str, object], fallback_key: str = "") -> str:
        for key in ("filename", "file_name", "file", "nama_file", "path"):
            val = entry_obj.get(key)
            if isinstance(val, str) and val.strip():
                return Path(val.strip()).name
        return Path(fallback_key.strip()).name if fallback_key.strip() else ""

    def add_entry(entry_obj: object, fallback_key: str = "") -> None:
        if not isinstance(entry_obj, dict):
            return
        filename = extract_filename(entry_obj, fallback_key=fallback_key)
        if not filename:
            return
        metadata_map[filename] = {
            "title": str(entry_obj.get("title") or Path(filename).stem),
            "category": str(entry_obj.get("category") or "Umum"),
            "source_url": str(entry_obj.get("source_url") or ""),
        }

    if isinstance(raw_data, list):
        for entry in raw_data:
            add_entry(entry)
    elif isinstance(raw_data, dict):
        if any(k in raw_data for k in ("filename", "file_name", "file", "nama_file")):
            add_entry(raw_data)
        else:
            # Cek apakah ada properti bernilai list (mis. {"documents": [...]}, {"dokumen": [...]})
            list_found = False
            for value in raw_data.values():
                if isinstance(value, list):
                    list_found = True
                    for entry in value:
                        add_entry(entry)

            if not list_found:
                # Format dict keyed by filename: {"nama_file.pdf": {"title": ...}}
                for fname_key, entry in raw_data.items():
                    add_entry(entry, fallback_key=str(fname_key))
    else:
        logger.error(
            "Isi %s harus berupa JSON array atau JSON object yang valid.",
            metadata_path,
        )
        sys.exit(1)

    logger.info("Berhasil memuat %d entri metadata dari %s.", len(metadata_map), metadata_path)
    return metadata_map


def read_pdf_pages(pdf_path: Path) -> list[Document]:
    """Membaca file PDF halaman per halaman dan mengekstrak teks serta page_label."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error(
            "Library 'pypdf' belum terpasang. Jalankan:\n"
            "pip install pypdf llama-index-readers-file"
        )
        sys.exit(1)

    reader = PdfReader(str(pdf_path))
    page_docs: list[Document] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        page_docs.append(
            Document(
                text=text,
                metadata={
                    "file_name": pdf_path.name,
                    "file_path": str(pdf_path),
                    "page_label": str(page_idx),
                },
            )
        )
    return page_docs


def read_txt_file(txt_path: Path) -> list[Document]:
    """Membaca file .txt UTF-8 dengan fallback encoding."""
    try:
        text = txt_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        text = txt_path.read_text(encoding="latin-1", errors="ignore").strip()

    if not text:
        return []

    return [
        Document(
            text=text,
            metadata={
                "file_name": txt_path.name,
                "file_path": str(txt_path),
                "page_label": "1",
            },
        )
    ]


def load_and_enrich_documents(
    raw_dir: Path, metadata_map: dict[str, dict[str, str]]
) -> list[Document]:
    """Memuat seluruh file .pdf dan .txt dari data/raw/ per halaman dan menyuntikkan metadata."""
    if not raw_dir.exists():
        logger.error("Direktori sumber dokumen tidak ditemukan: %s", raw_dir)
        sys.exit(1)

    supported_files = sorted(
        p
        for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}
    )
    if not supported_files:
        logger.error(
            "Tidak ada file .pdf atau .txt di %s. Tambahkan dokumen terlebih dahulu.",
            raw_dir,
        )
        sys.exit(1)

    logger.info(
        "Ditemukan %d file (.pdf/.txt) di %s. Memulai ekstraksi teks per halaman...",
        len(supported_files),
        raw_dir,
    )

    documents: list[Document] = []
    missing_metadata_files: set[str] = set()

    for file_path in supported_files:
        logger.info("Membaca file: %s...", file_path.name)
        if file_path.suffix.lower() == ".pdf":
            file_docs = read_pdf_pages(file_path)
        else:
            file_docs = read_txt_file(file_path)

        if not file_docs:
            logger.warning("Tidak ada teks yang dapat diekstrak dari %s (dilewati).", file_path.name)
            continue

        doc_meta = metadata_map.get(file_path.name)
        if doc_meta is None:
            missing_metadata_files.add(file_path.name)
            doc_meta = {
                "title": file_path.stem.replace("_", " ").replace("-", " ").title(),
                "category": "Umum",
                "source_url": "",
            }

        for doc in file_docs:
            doc.metadata["file_name"] = file_path.name
            doc.metadata["title"] = doc_meta["title"]
            doc.metadata["category"] = doc_meta["category"]
            doc.metadata["source_url"] = doc_meta["source_url"]
            doc.metadata["page_label"] = str(doc.metadata.get("page_label") or "1")
            documents.append(doc)

        logger.info("  -> %d halaman/bagian berhasil diekstrak dari %s.", len(file_docs), file_path.name)

    if not documents:
        logger.error("Tidak ada konten teks yang berhasil diekstrak dari seluruh dokumen.")
        sys.exit(1)

    if missing_metadata_files:
        logger.warning(
            "File berikut tidak terdaftar di metadata.json (memakai nilai default): %s",
            ", ".join(sorted(missing_metadata_files)),
        )

    logger.info("Total %d halaman/dokumen siap dipecah menjadi nodes.", len(documents))
    return documents


def split_documents_into_nodes(documents: list[Document]) -> list[BaseNode]:
    """Memecah dokumen menjadi chunk menggunakan SentenceSplitter dan memastikan metadata lengkap."""
    logger.info(
        "Memecah teks dengan SentenceSplitter (chunk_size=%d, chunk_overlap=%d)...",
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

    # Tetapkan ID deterministik agar re-indexing lebih rapi & verifikasi field metadata wajib
    for idx, node in enumerate(nodes):
        file_name = str(node.metadata.get("file_name", "unknown"))
        page_label = str(node.metadata.get("page_label", "1"))
        content_hash = hashlib.sha1(node.get_content().encode("utf-8")).hexdigest()[:12]
        node.id_ = f"{file_name}::p{page_label}::{idx}::{content_hash}"

        node.metadata.setdefault("title", Path(file_name).stem)
        node.metadata.setdefault("category", "Umum")
        node.metadata.setdefault("source_url", "")
        node.metadata["page_label"] = page_label

    logger.info("Berhasil menghasilkan %d text nodes.", len(nodes))
    return nodes


def index_nodes_to_chroma(
    nodes: list[BaseNode],
    embed_model: GoogleGenAIEmbedding,
    model_name: str,
    chroma_dir: Path,
    collection_name: str,
    reset_collection: bool = False,
) -> None:
    """Mengindeks nodes ke ChromaDB persisten secara bertahap dengan retry rate limit."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Menghubungkan ke ChromaDB persisten di %s...", chroma_dir)

    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    if reset_collection:
        existing_names = [col.name for col in chroma_client.list_collections()]
        if collection_name in existing_names:
            logger.info("Menghapus koleksi lama '%s' (--reset aktif)...", collection_name)
            chroma_client.delete_collection(name=collection_name)

    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes=[],
        storage_context=storage_context,
        embed_model=embed_model,
    )

    total_nodes = len(nodes)
    total_batches = (total_nodes + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(
        "Memulai embedding (%s) & penyimpanan ke koleksi '%s' (%d nodes dalam %d batch)...",
        model_name,
        collection_name,
        total_nodes,
        total_batches,
    )

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, total_nodes)
        batch_nodes = nodes[start:end]

        retry_with_backoff(
            func=lambda b=batch_nodes: index.insert_nodes(b),
            action_label=f"embedding & indexing batch {batch_idx + 1}/{total_batches}",
        )
        logger.info(
            "Progres indexing: batch %d/%d selesai (%d/%d nodes tersimpan).",
            batch_idx + 1,
            total_batches,
            end,
            total_nodes,
        )

    logger.info(
        "Indexing ChromaDB selesai. Total vektor di koleksi '%s': %d.",
        collection_name,
        chroma_collection.count(),
    )


def save_nodes_pickle(nodes: list[BaseNode], pickle_path: Path) -> None:
    """Menyimpan daftar nodes ke file pickle untuk kebutuhan BM25Retriever."""
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with pickle_path.open("wb") as file:
        pickle.dump(nodes, file, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Berhasil menyimpan %d nodes ke %s.", len(nodes), pickle_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest dokumen pastoral & liturgi Katolik ke ChromaDB dan nodes.pkl."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Hapus koleksi ChromaDB yang sudah ada sebelum melakukan indexing ulang.",
    )
    return parser.parse_args()


def main() -> None:
    start_time = time.time()
    args = parse_args()

    logger.info("=== Memulai Pipeline Ingest RAG Pastoral & Liturgi Katolik ===")

    api_key = load_api_key()
    embed_model, active_embedding_model = create_embedding_model(api_key)

    metadata_map = load_metadata_map(METADATA_FILE)
    documents = load_and_enrich_documents(RAW_DATA_DIR, metadata_map)
    nodes = split_documents_into_nodes(documents)
    save_nodes_pickle(nodes, NODES_PICKLE_PATH)

    index_nodes_to_chroma(
        nodes=nodes,
        embed_model=embed_model,
        model_name=active_embedding_model,
        chroma_dir=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME,
        reset_collection=args.reset,
    )

    elapsed = time.time() - start_time
    logger.info("=== Pipeline Ingest Selesai dalam %.2f detik ===", elapsed)


if __name__ == "__main__":
    main()