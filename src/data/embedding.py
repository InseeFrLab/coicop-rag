# %%
import os
import duckdb

con = duckdb.connect(database=":memory:")

s3_path = "s3://projet-budget-famille/data/coicop-2018_envoi_rmes_20251022.csv"
query = f"""
    SELECT
        * 
    FROM read_csv('{s3_path}');
"""
notices_raw = duckdb.sql(query).to_df()

columns_to_keep = [
    col for col in notices_raw.columns 
    if 'column' not in col.lower() and not col.endswith('_en')
]

notices_raw = notices_raw[columns_to_keep]

# %%
notices_raw
notices_raw.loc[notices_raw["type"] == "Poste"].isna().sum()

# %%
from dataclasses import dataclass
import pandas as pd
from typing import List, Dict, Any, Optional
@dataclass
from dataclasses import dataclass
from typing import Optional, List, Dict

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
    
    def to_text_chunks(self, strategy: str = "all_info") -> List[Dict[str, str]]:
        """
        Convert to text chunks for embedding according to strategy.
        strategy: "code_only", "all_info", "all_info_no_exclusions"
        """
        chunk = {
            "type": "title",
            "text": self.to_single_text(strategy),
            "code": self.code
        }
        return chunk  
        
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

# %%
df = notices_raw[:10]
doc = []
for _, row in df.iterrows():
        doc.append(
            CoicopDocument(
                code=str(row['code']),
                label_fr=str(row['label_fr']),
                note_generale_fr=row.get('note_generale_fr'),
                contenu_central_fr=row.get('contenu_central_fr'),
                contenu_additionnel_fr=row.get('contenu_additionnel_fr'),
                note_exclusion_fr=row.get('note_exclusion_fr')
            )
        )

# %%
print(doc[6].to_text_chunks(strategy="code_only")["text"])
print(doc[6].to_text_chunks(strategy="without_exclusions")["text"])
print(doc[6].to_text_chunks()["text"])
print(doc[6].to_single_text())
