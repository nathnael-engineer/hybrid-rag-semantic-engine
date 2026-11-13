from dataclasses import dataclass
import hashlib
import re
from typing import List

@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    parent_id: str
    text: str
    section_header: str
    is_table: bool

def _hash_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

def layout_aware_chunk(
    doc_id: str,
    raw_text: str,
    parent_size: int = 1200,
    child_size: int = 360,
    overlap: int = 64,
) -> List[DocumentChunk]:
    blocks = re.split(r"\n(?=#{1,3}\s)", raw_text.strip())
    chunks: List[DocumentChunk] = []

    for b_idx, block in enumerate(blocks):
        lines = block.splitlines()
        header = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else "General"
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else block

        if "|" in body and "---" in body:
            parent_id = _hash_id(f"{doc_id}:p:{b_idx}")
            chunks.append(
                DocumentChunk(
                    chunk_id=_hash_id(f"{parent_id}:table"),
                    doc_id=doc_id,
                    parent_id=parent_id,
                    text=body,
                    section_header=header,
                    is_table=True,
                )
            )
            continue

        for p_start in range(0, max(1, len(body)), parent_size):
            parent_text = body[p_start : p_start + parent_size]
            parent_id = _hash_id(f"{doc_id}:p:{b_idx}:{p_start}")
            step = max(1, child_size - overlap)
            for c_start in range(0, max(1, len(parent_text)), step):
                child_text = parent_text[c_start : c_start + child_size].strip()
                if not child_text:
                    continue
                chunks.append(
                    DocumentChunk(
                        chunk_id=_hash_id(f"{parent_id}:c:{c_start}"),
                        doc_id=doc_id,
                        parent_id=parent_id,
                        text=f"[{header}] {child_text}",
                        section_header=header,
                        is_table=False,
                    )
                )
    return chunks
