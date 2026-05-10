from typing import List
from pydantic import BaseModel, Field

class RewrittenQuestion(BaseModel):
    rewritten_question: str = Field(
        description="The clarified and rephrased version of the user's question, rewritten for maximum clarity based on conversation context."
    )

class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities  that " "appear in the text",
    )

class DocumentSummary(BaseModel):
    brief_overview: str = Field(
        description="A brief summary (2-3 sentences) that describes the entire content of the document."
    )
    detailed_summary: str = Field(
        description="A detailed summary covering all information present in the document."
    )

class Chunk(BaseModel):
    title: str = Field(description="A short, descriptive title for this chunk")
    content: str = Field(description="The text content of this chunk")
    keywords: List[str] = Field(description="The keywords extracted from the content of this chunk")

class ChunkedSummary(BaseModel):
    chunks: List[Chunk] = Field(
        description="A list of semantically meaningful chunks, each with a title and content"
    )

class GraphNode(BaseModel):
    id: str = Field(description="Canonical identifier of the node (e.g. the entity name, normalized).")
    type: str = Field(description="Node label/type, must be one of the allowed node types.")
    properties: dict = Field(default_factory=dict, description="Optional key/value properties for the node.")

class GraphRelationship(BaseModel):
    source_id: str = Field(description="Canonical id of the source node.")
    source_type: str = Field(description="Label/type of the source node.")
    target_id: str = Field(description="Canonical id of the target node.")
    target_type: str = Field(description="Label/type of the target node.")
    type: str = Field(description="Relationship type in UPPER_SNAKE_CASE.")
    properties: dict = Field(default_factory=dict, description="Optional key/value properties for the relationship.")

class GraphExtraction(BaseModel):
    nodes: List[GraphNode] = Field(description="Entities extracted from the text.")
    relationships: List[GraphRelationship] = Field(description="Relationships among the extracted entities.")
