"""
DynamicSectionRetrieverIngestion.py — A-RAG document ingestion pipeline.

Output artefacts
----------------
chunks.json  (Config.CHUNKS_FILE)
    [{"id": "0", "text": "...", "section_id": "...", "sub_section_id": "...",
      "page_num": 1, "paper_path": "..."}, ...]

index/sentence_index.pkl  (Config.INDEX_DIR)
    {
      "sentences"         : List[str],        # sentence strings
      "embeddings"        : np.ndarray(N×D),  # L2-normalised, float32
      "sentence_to_chunk" : List[str],        # parallel chunk_id per sentence
      "chunks"            : Dict[str, Dict],  # {chunk_id: {text, ...}}
    }

Design (arXiv:2602.03442, Section 3.1):
- LlamaParse produces page-level markdown nodes.
- GPT-4o-mini extracts section metadata from each page.
- SentenceSplitter re-chunks at ~1000 tokens with 100-token overlap.
- sentence-transformers builds the local sentence embedding index
  (no OpenAI embedding cost at inference time).
"""

import asyncio
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field
from llama_index.core.schema import TextNode
from llama_index.core.prompts import ChatMessage
from llama_index.core.async_utils import run_jobs
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is required. Install with: pip install sentence-transformers"
    )


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------

class SectionOutput(BaseModel):
    section_name: str = Field(..., description="Section number, e.g. '3.2'")
    section_title: str = Field(..., description="Section title text")
    start_page_number: int = Field(..., description="Page where this section begins")
    is_subsection: bool = Field(..., description="True when this is a sub-section")
    description: Optional[str] = Field(None, description="Verbatim extracted line")

    def get_section_id(self) -> str:
        return f"{self.section_name}: {self.section_title}"


class SectionsOutput(BaseModel):
    sections: List[SectionOutput]


class ValidSections(BaseModel):
    valid_indexes: List[int] = Field(description="Indexes of valid sections.")


# ---------------------------------------------------------------------------
# Document Processor
# ---------------------------------------------------------------------------

class DocumentProcessor:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100   # 10 % of chunk size

    def __init__(
        self,
        llama_cloud_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_device: Optional[str] = None,
    ):
        self.llama_cloud_api_key = llama_cloud_api_key or os.getenv("LLAMA_CLOUD_API_KEY", "")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")

        self.llm = OpenAI(
            model="gpt-4o-mini", api_key=self.openai_api_key, temperature=0.1
        )
        Settings.llm = self.llm
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-large", api_key=self.openai_api_key
        )

        self.parser = LlamaParse(
            parse_mode="parse_page_with_agent",
            model="openai-gpt-4o-mini",
            api_key=self.llama_cloud_api_key,
            result_type="markdown",
            high_res_ocr=True,
            adaptive_long_table=True,
            outlined_table_extraction=True,
            output_tables_as_HTML=True,
        )

        self.chunk_splitter = SentenceSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
        )
        # Small splitter: produces ~1–2 sentence nodes for the sentence index.
        self.sentence_splitter = SentenceSplitter(chunk_size=64, chunk_overlap=0)

        print(f"Loading sentence embedding model '{embedding_model}'…")
        self.sentence_model = SentenceTransformer(embedding_model, device=embedding_device)
        print("Embedding model ready.")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def process_documents(
        self,
        file_paths: List[str],
        chunks_file: str = "chunks.json",
        index_dir: str = "index",
    ) -> None:
        if not file_paths:
            raise ValueError("file_paths must not be empty.")

        # 1. Parse documents
        print(f"\nParsing {len(file_paths)} document(s) with LlamaParse…")
        results = await self.parser.aparse(file_paths)

        # 2. Page-level nodes
        print("Building page-level nodes…")
        page_nodes_dict: Dict[str, List[TextNode]] = {}
        for i, result in enumerate(results):
            nodes = self._get_page_nodes(result, file_paths[i])
            page_nodes_dict[file_paths[i]] = nodes
            print(f"  {Path(file_paths[i]).name}: {len(nodes)} pages")

        # 3. Section extraction
        print("Extracting section structure (LLM)…")
        sections_dict = await self._acreate_sections(page_nodes_dict)

        # 4. Annotate pages
        print("Annotating pages with section metadata…")
        for fp, nodes in page_nodes_dict.items():
            self._annotate_pages_with_sections(nodes, sections_dict.get(fp, []))

        # 5. Build chunks + sentences
        print("Re-chunking and extracting sentences…")
        chunks, sentences, sent_to_chunk = self._build_chunks_and_sentences(page_nodes_dict)
        print(f"  Chunks: {len(chunks)} | Sentences: {len(sentences)}")

        # 6. Save chunks.json
        out_path = Path(chunks_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_list = [{"id": cid, **meta} for cid, meta in chunks.items()]
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(chunk_list, f, ensure_ascii=False, indent=2)
        print(f"Saved chunks.json → {out_path}")

        # 7. Encode sentences and save pickle index
        print("Encoding sentences (sentence-transformers)…")
        embeddings = self.sentence_model.encode(
            sentences,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=128,
        ).astype(np.float32)

        idx_dir = Path(index_dir)
        idx_dir.mkdir(parents=True, exist_ok=True)
        idx_file = idx_dir / "sentence_index.pkl"
        with idx_file.open("wb") as f:
            pickle.dump(
                {
                    "sentences": sentences,
                    "embeddings": embeddings,
                    "sentence_to_chunk": sent_to_chunk,
                    "chunks": chunks,
                },
                f,
            )
        print(f"Saved sentence_index.pkl → {idx_file}")
        print("\n✅ Ingestion complete.")

    # ------------------------------------------------------------------
    # Core node construction
    # ------------------------------------------------------------------

    def _build_chunks_and_sentences(
        self, page_nodes_dict: Dict[str, List[TextNode]]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
        chunks: Dict[str, Dict[str, Any]] = {}
        sentences: List[str] = []
        sentence_to_chunk: List[str] = []
        global_idx = 0

        for file_path, page_nodes in page_nodes_dict.items():
            docs = [
                Document(
                    text=n.get_content(metadata_mode="none"),
                    metadata={**n.metadata},
                )
                for n in page_nodes
            ]
            for raw_node in self.chunk_splitter.get_nodes_from_documents(docs):
                chunk_id = str(global_idx)
                global_idx += 1

                chunk_text = raw_node.get_content(metadata_mode="none")
                chunks[chunk_id] = {
                    "text": chunk_text,
                    "section_id": raw_node.metadata.get("section_id", ""),
                    "sub_section_id": raw_node.metadata.get("sub_section_id", ""),
                    "page_num": raw_node.metadata.get("page_num", 0),
                    "paper_path": file_path,
                }

                # Sentence-level nodes for the embedding index
                for sn in self.sentence_splitter.get_nodes_from_documents(
                    [Document(text=chunk_text)]
                ):
                    s = sn.get_content(metadata_mode="none").strip()
                    if len(s) >= 10:
                        sentences.append(s)
                        sentence_to_chunk.append(chunk_id)

        return chunks, sentences, sentence_to_chunk

    # ------------------------------------------------------------------
    # Page-level node creation
    # ------------------------------------------------------------------

    def _get_page_nodes(self, result, file_path: str) -> List[TextNode]:
        nodes: List[TextNode] = []
        pages = getattr(result, "pages", [])
        if not pages and isinstance(result, list):
            pages = result
        for idx, page in enumerate(pages):
            text = getattr(page, "md", getattr(page, "text", str(page)))
            nodes.append(
                TextNode(
                    text=text,
                    metadata={"page_num": idx + 1, "paper_path": file_path},
                )
            )
        return nodes

    # ------------------------------------------------------------------
    # Section extraction
    # ------------------------------------------------------------------

    async def _aget_sections(self, doc_text: str) -> List[SectionOutput]:
        system_prompt = (
            "Extract section metadata from the document text.\n"
            "- Only extract if the text clearly starts a new section.\n"
            "- A valid section MUST begin with '#' and contain a number.\n"
            "- Do NOT extract figures, tables, or captions as sections."
        )
        sllm = self.llm.as_structured_llm(SectionsOutput)
        messages = [
            ChatMessage(content=system_prompt, role="system"),
            ChatMessage(content=f"Document text:\n\n{doc_text}", role="user"),
        ]
        try:
            result = await sllm.achat(messages)
            return result.raw.sections
        except Exception:
            return []

    async def _arefine_sections(
        self, sections: List[SectionOutput]
    ) -> List[SectionOutput]:
        if not sections:
            return []
        system_prompt = (
            "Review the extracted sections and return only the valid indexes.\n"
            "Remove false positives based on sequential ordering."
        )
        sllm = self.llm.as_structured_llm(ValidSections)
        section_texts = "\n".join(
            f"{i}: {s.model_dump_json()}" for i, s in enumerate(sections)
        )
        messages = [
            ChatMessage(content=system_prompt, role="system"),
            ChatMessage(content=f"Sections:\n\n{section_texts}", role="user"),
        ]
        try:
            result = await sllm.achat(messages)
            valid = set(result.raw.valid_indexes)
            return [s for i, s in enumerate(sections) if i in valid]
        except Exception:
            return sections

    async def _acreate_sections(
        self, page_nodes_dict: Dict[str, List[TextNode]]
    ) -> Dict[str, List[SectionOutput]]:
        out: Dict[str, List[SectionOutput]] = {}
        for paper_path, nodes in page_nodes_dict.items():
            tasks = [
                self._aget_sections(n.get_content(metadata_mode="all"))
                for n in nodes
            ]
            raw_results = await run_jobs(tasks, workers=5, show_progress=True)
            all_secs = [s for r in raw_results for s in r]
            refined = await self._arefine_sections(all_secs)
            out[paper_path] = refined
            print(f"  {Path(paper_path).name}: {len(refined)} sections found")
        return out

    # ------------------------------------------------------------------
    # Section annotation
    # ------------------------------------------------------------------

    def _annotate_pages_with_sections(
        self, nodes: List[TextNode], sections: List[SectionOutput]
    ) -> None:
        if not sections:
            return
        main_secs = [s for s in sections if not s.is_subsection]
        all_secs = sections
        mi = si = 0
        for node in nodes:
            cur = node.metadata.get("page_num", 0)
            while mi + 1 < len(main_secs) and main_secs[mi + 1].start_page_number <= cur:
                mi += 1
            while si + 1 < len(all_secs) and all_secs[si + 1].start_page_number <= cur:
                si += 1
            if main_secs:
                node.metadata["section_id"] = main_secs[mi].get_section_id()
            if all_secs:
                node.metadata["sub_section_id"] = all_secs[si].get_section_id()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    try:
        from config import Config
        chunks_file = Config.CHUNKS_FILE
        index_dir = Config.INDEX_DIR
    except (ImportError, AttributeError):
        chunks_file = "chunks.json"
        index_dir = "index"

    data_dir = Path("./data")
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        print(f"Created {data_dir}. Place PDF files there and re-run.")
        return

    files = sorted(str(p) for p in data_dir.iterdir() if p.suffix.lower() == ".pdf")
    if not files:
        print(f"No PDF files found in {data_dir}.")
        return

    print(f"Found {len(files)} PDF(s): {[Path(f).name for f in files]}")
    processor = DocumentProcessor()
    await processor.process_documents(
        file_paths=files,
        chunks_file=chunks_file,
        index_dir=index_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())
