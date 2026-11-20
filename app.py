import sys
from pathlib import Path
from typing import Any

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.rag_engine import ask_question  # noqa: E402

# 1. Konfigurasi halaman
st.set_page_config(
    page_title="Asisten Praktis Pastoral & Liturgi Katolik",
    page_icon="✝️",
    layout="wide",
)

SUPPORTED_TOPICS = [
    "Perkawinan & Sakramen",
    "Pendampingan Orang Sakit & Kematian",
    "Benda Rohani",
    "Krisis & Moral",
]

DEFAULT_WELCOME_MESSAGE = (
    "Halo! Saya adalah **Asisten Praktis Pastoral & Liturgi Gereja Katolik**.\n\n"
    "Silakan ajukan pertanyaan seputar praktik pastoral, tata cara liturgi, sakramen, "
    "maupun pedoman moral berdasarkan dokumen resmi Gereja yang telah diindeks."
)


def init_session_state() -> None:
    """Inisialisasi riwayat chat pada session_state."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": DEFAULT_WELCOME_MESSAGE,
                "sources": [],
                "is_welcome": True,
            }
        ]


def reset_chat_history() -> None:
    """Menghapus riwayat percakapan dan mengembalikan pesan pembuka."""
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": DEFAULT_WELCOME_MESSAGE,
            "sources": [],
            "is_welcome": True,
        }
    ]


def render_sources_expander(sources: list[dict[str, Any]]) -> None:
    """Merender daftar sumber dokumen rujukan dan tautan asli di bawah jawaban asisten."""
    with st.expander("📚 Lihat Sumber Dokumen & Tautan Asli"):
        if not sources:
            st.caption("Tidak ada sumber rujukan dokumen yang cocok untuk jawaban ini.")
            return

        for idx, src in enumerate(sources, start=1):
            title = str(src.get("title") or "Dokumen Tanpa Judul")
            category = str(src.get("category") or "").strip()
            page_label = str(src.get("page_label") or "1")
            source_url = str(src.get("source_url") or "").strip()
            snippet = str(src.get("snippet") or "").strip()

            category_label = f" | **Kategori:** {category}" if category else ""
            st.markdown(
                f"**{idx}. {title}**  \n"
                f"**Halaman:** {page_label}{category_label}"
            )

            if source_url:
                st.markdown(f"[Buka Dokumen Asli]({source_url})")
            else:
                st.caption("Tautan dokumen asli belum tersedia.")

            if snippet:
                st.info(f'"{snippet}..."')

            if idx < len(sources):
                st.divider()


def render_sidebar() -> None:
    """Merender sidebar kiri berisi deskripsi, pilar topik, dan tombol reset chat."""
    with st.sidebar:
        st.header("✝️ Asisten Pastoral & Liturgi")
        st.markdown(
            "**Deskripsi Aplikasi**  \n"
            "Asisten tanya-jawab berbasis *Retrieval-Augmented Generation* (RAG) untuk "
            "membantu pelayan pastoral, katekis, dan umat menemukan panduan praktis "
            "pastoral serta liturgi Gereja Katolik secara cepat dan terstruktur."
        )

        st.warning(
            "**Batasan Sistem:** Jawaban hanya bersumber dari dokumen resmi Gereja Katolik "
            "yang tersimpan di basis data lokal (Katekismus, Kitab Hukum Kanonik, dan "
            "Dokumen Gerejawi). Sistem tidak menggantikan bimbingan langsung pastor paroki "
            "atau otoritas keuskupan."
        )

        st.subheader("Pilar Topik yang Didukung")
        for topic in SUPPORTED_TOPICS:
            st.markdown(f"- {topic}")

        st.divider()
        if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True, type="secondary"):
            reset_chat_history()
            st.rerun()


def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("Asisten Praktis Pastoral & Liturgi Katolik")
    st.caption(
        "Ajukan pertanyaan praktis seputar sakramen, pendampingan pastoral, "
        "liturgi, dan ajaran resmi Gereja Katolik."
    )

    # Tampilkan seluruh riwayat percakapan
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and not message.get("is_welcome", False):
                render_sources_expander(message.get("sources", []))

    # Input pertanyaan pengguna
    if prompt := st.chat_input("Tulis pertanyaan pastoral atau liturgi Anda di sini..."):
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            return

        # Simpan dan tampilkan pesan pengguna
        st.session_state.messages.append(
            {
                "role": "user",
                "content": cleaned_prompt,
            }
        )
        with st.chat_message("user"):
            st.markdown(cleaned_prompt)

        # Proses jawaban melalui Hybrid RAG Engine
        with st.chat_message("assistant"):
            try:
                with st.spinner("Mencari rujukan dokumen resmi..."):
                    result = ask_question(cleaned_prompt)

                answer = str(result.get("answer") or "").strip()
                sources = result.get("sources") or []

                st.markdown(answer)
                render_sources_expander(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "is_welcome": False,
                    }
                )
            except (FileNotFoundError, ValueError) as exc:
                error_msg = (
                    f"Basis data dokumen belum siap: {exc}\n\n"
                    "Pastikan Anda sudah menjalankan `python src/ingest.py` terlebih dahulu."
                )
                st.error(error_msg)
            except (EnvironmentError, PermissionError) as exc:
                st.error(f"Kendala konfigurasi API Key Gemini: {exc}")
            except RuntimeError as exc:
                st.error(
                    f"Batas permintaan (rate limit) API tercapai: {exc}\n\n"
                    "Silakan tunggu beberapa saat sebelum mengirim pertanyaan berikutnya."
                )
            except Exception as exc:
                st.error(f"Terjadi kesalahan tak terduga saat memproses pertanyaan: {exc}")


if __name__ == "__main__":
    main()