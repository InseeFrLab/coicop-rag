from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class CoicopDocument:
    """Structure of a COICOP document to embed"""
    code: str
    label_fr: str
    note_generale_fr: Optional[str] = None
    contenu_central_fr: Optional[str] = None
    contenu_additionnel_fr: Optional[str] = None
    note_exclusion_fr: Optional[str] = None
    
    @property
    def inclusions(self) -> Optional[str]:
        """Concatenate central and additional content"""
        parts = []
        if self.contenu_central_fr:
            parts.append(self.contenu_central_fr.strip())
        if self.contenu_additionnel_fr:
            parts.append(self.contenu_additionnel_fr.strip())
        return ". ".join(parts) if parts else None
        
    def to_single_text(self, strategy: str = "all_info") -> str:
        """
        Convert to a single Markdown-formatted text according to strategy.
        Each section is separated by a clear line break for embedding.
        """
        lines = [f"**{self.code}: {self.label_fr}**"]  # Bold title
        
        if strategy != "code_only":
            if self.note_generale_fr:
                lines.append(f"**General note:** {self.note_generale_fr}")
            
            inclusions = self.inclusions
            if inclusions:
                lines.append(f"**Inclusions:** {inclusions}")
            
            if strategy == "all_info" and self.note_exclusion_fr:
                lines.append(f"**Exclusions:** {self.note_exclusion_fr}")
        
        # Join sections with two line breaks for clear separation in embeddings
        return "\n\n".join(lines)
    
    def to_text_chunks(self, strategy: str = "all_info") -> List[Dict[str, str]]:
        """
        Convert to text chunks for embedding according to strategy.
        """
        chunk = {
            "type": strategy,
            "text": self.to_single_text(strategy),
            "code": self.code
        }
        return chunk