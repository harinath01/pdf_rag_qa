from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from bs4 import BeautifulSoup

from marker.renderers.json import JSONOutput, JSONBlockOutput
from marker.schema import BlockTypes
from chonkie.chunker.semantic import SemanticChunker

class ChunkType(str, Enum):
    SECTION = "section"
    FOOTNOTE = "footnote"
    TABLE = "table"
    PARAGRAPH = "paragraph"
    
class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    
    def __eq__(self, other):
        return self.x0 == other.x0 and self.y0 == other.y0 and self.x1 == other.x1 and self.y1 == other.y1
    
    def __str__(self):
        return f"BoundingBox(x0={self.x0}, y0={self.y0}, x1={self.x1}, y1={self.y1})"

class Citation(BaseModel):
    bbox: BoundingBox
    page: int
    
    def __eq__(self, other):
        return self.bbox == other.bbox and self.page == other.page
    
    def __str__(self):
        return f"Citation(bbox={self.bbox}, page={self.page})"

class Title(BaseModel):
    text: str
    bbox: BoundingBox
    page: int
    level: int

class Content(BaseModel):
    text: str
    citations: List[Citation]
    token_count: int = Field(default=0)


class Chunk(BaseModel):
    chunk_id: str
    title: Optional[Title]
    parent_title: Optional[Title]
    content: Content
    type: ChunkType
    metadata: Dict[str, Any] = {}
     
    def __str__(self):
        base_str = ""
        if self.title:
            base_str += f"\nTitle: {self.title.text}\n"
        if self.parent_title:
            base_str += f"Parent Title: {self.parent_title.text}\n"
        if self.content:
            base_str += f"Content: {self.content.text}\n"
        if self.type:
            base_str += f"Type: {self.type}\n"
        if self.metadata:
            base_str += f"Metadata: {self.metadata}\n"
        return base_str
        
        
def chunk_json_output(json_output: JSONOutput, max_chunk_size: int = 1000, overlap: int = 100) -> List[Chunk]:
    chunks = []
    current_chunk = None
    parent_titles = {}
    chunker = SemanticChunker(
        embedding_model="minishlab/potion-base-8M",
        threshold=0.5,
        chunk_size=max_chunk_size,
        min_sentences=1
    )
    
    for page in json_output.children:
        for block in page.children:
            if block.block_type == str(BlockTypes.SectionHeader):
                section_level = len(block.section_hierarchy) if block.section_hierarchy else 1
                
                # Find the closest parent level (the highest level that's less than current)
                parent_level = max([level for level in parent_titles.keys() if level < section_level], default=None)
                parent_title = parent_titles.get(parent_level) if parent_level is not None else None
                
                # Create new section chunk
                current_chunk = create_section_chunk(block, len(chunks)+1, get_page_number(block.id), parent_title)
                chunks.append(current_chunk)
                
                # Update parent titles
                parent_titles[section_level] = current_chunk.title
                
                # Clear any deeper level parents
                for level in list(parent_titles.keys()):
                    if level > section_level:
                        del parent_titles[level]
                        
            elif block.block_type == str(BlockTypes.Footnote):
                current_chunk = create_footnote_chunk(block, len(chunks)+1)
                chunks.append(current_chunk)
            elif block.block_type in [str(BlockTypes.Text), str(BlockTypes.ListItem)] and current_chunk:
                # Calculate new token count before appending
                new_text = current_chunk.content.text + " " + parse_html_text(block.html)
                new_token_count = count_tokens(new_text)
                
                # Check if we need to split the content
                if new_token_count > max_chunk_size:
                    # Use SemanticChunker to split content
                    semantic_chunks = chunker.chunk(new_text)
                    
                    # Create new chunks for each semantic chunk
                    for i, semantic_chunk in enumerate(semantic_chunks):
                        if i == 0:
                            # Update current chunk with first semantic chunk
                            current_chunk.content.text = semantic_chunk.text
                            current_chunk.content.token_count = semantic_chunk.token_count
                        else:
                            # Create new chunk for remaining semantic chunks
                            new_chunk = create_section_chunk(
                                text=semantic_chunk.text,
                                bbox=block.bbox,
                                page_number=get_page_number(block.id),
                                chunk_type=current_chunk.type,
                                title=current_chunk.title,
                                parent_title=current_chunk.parent_title
                            )
                            chunks.append(new_chunk)
                            current_chunk = new_chunk
                else:
                    # If within token limit, append content
                    current_chunk.content.text = new_text
                    current_chunk.content.token_count = new_token_count
                    
                    # Add new citation if it doesn't exist
                    new_citation = Citation(
                        bbox=BoundingBox(**convert_bbox_to_dict(block.bbox)),
                        page=get_page_number(block.id)
                    )
                    if new_citation not in current_chunk.content.citations:
                        current_chunk.content.citations.append(new_citation)
            
    return chunks
            

def create_section_chunk(block: JSONBlockOutput, chunk_counter: int, level: int, parent_title: Optional[Title] = None) -> Chunk:
    return Chunk(
        chunk_id=f"chunk_{chunk_counter:03d}",
        title=Title(
            text=parse_html_text(block.html),
            bbox=BoundingBox(**convert_bbox_to_dict(block.bbox)),
            page=get_page_number(block.id),
            level=level
        ),
        parent_title=parent_title, 
        content=Content(
            text="",
            citations=[],  # Initialize empty citations
            token_count=0
        ),
        type=ChunkType.SECTION,
    )

def create_footnote_chunk(block: JSONBlockOutput, chunk_counter: int) -> Chunk:
    content_text = parse_html_text(block.html)
    return Chunk(
        chunk_id=f"chunk_{chunk_counter:03d}",
        title=None,
        parent_title=None,
        content=Content(
            text=content_text,
            citations=[Citation(
                bbox=BoundingBox(**convert_bbox_to_dict(block.bbox)),
                page=get_page_number(block.id)
            )],
            token_count=count_tokens(content_text)
        ),
        type=ChunkType.FOOTNOTE
    )


    
def convert_bbox_to_dict(bbox: List[float]) -> Dict[str, float]:
    if len(bbox) != 4:
        raise ValueError("Bounding box must contain exactly 4 values [x0, y0, x1, y1]")
    
    return {
        "x0": bbox[0],
        "y0": bbox[1],
        "x1": bbox[2],
        "y1": bbox[3]
    }
    
def get_page_number(id: str) -> int:
    return int(id.split("page/")[-1].split("/")[0])

def count_tokens(text: str) -> int:
    return len(text.split(" "))

def parse_html_text(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    
    return soup.get_text(separator=' ', strip=True)