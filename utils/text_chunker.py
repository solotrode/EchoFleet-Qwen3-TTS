import re
from typing import List


def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """Split `text` into chunks by sentence boundaries with hard limits.

    Behavior:
    - Default maximum characters per chunk is `max_chars`.
    - Splits on sentence boundaries (ends with ., !, ? followed by whitespace or end).
    - Builds chunks by accumulating whole sentences until adding the next would exceed
      `max_chars`, then starts a new chunk.
    - If a single sentence exceeds `max_chars * 1.5`, it will be split mid-sentence
      at word boundaries to prevent extremely long chunks.

    Args:
        text: Input text to split.
        max_chars: Preferred maximum number of characters per chunk.

    Returns:
        A list of text chunks (strings).
    """
    if not text:
        return []

    # Normalize whitespace
    normalized = " ".join(text.split())

    # Split into sentences using a more robust regex that handles end-of-string
    # This matches: period/question/exclamation followed by space OR end of string
    sentence_end_re = re.compile(r'(?<=[.!?])(?:\s+|$)')
    raw_sentences = sentence_end_re.split(normalized)
    
    # Filter out empty strings and strip whitespace
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    
    # Hard limit: if a single sentence is > max_chars * 1.5, split it
    hard_limit = int(max_chars * 1.5)

    for sent in sentences:
        sent_len = len(sent)
        
        # If this sentence is extremely long, split it at word boundaries
        if sent_len > hard_limit:
            # Flush current chunk if we have one
            if current:
                chunks.append(' '.join(current))
                current = []
                current_len = 0
            
            # Split the long sentence into word-boundary chunks
            words = sent.split()
            temp_chunk = []
            temp_len = 0
            
            for word in words:
                word_len = len(word)
                if temp_len + 1 + word_len > max_chars and temp_chunk:
                    # Flush this sub-chunk
                    chunks.append(' '.join(temp_chunk))
                    temp_chunk = [word]
                    temp_len = word_len
                else:
                    temp_chunk.append(word)
                    temp_len += (1 if temp_chunk else 0) + word_len
            
            if temp_chunk:
                chunks.append(' '.join(temp_chunk))
            
            continue

        # Normal sentence accumulation
        if not current:
            current.append(sent)
            current_len = sent_len
            continue

        # If adding this sentence would exceed max_chars, flush current and start new
        if current_len + 1 + sent_len > max_chars:
            chunks.append(' '.join(current))
            current = [sent]
            current_len = sent_len
        else:
            current.append(sent)
            current_len += 1 + sent_len  # account for separating space

    if current:
        chunks.append(' '.join(current))

    return chunks
