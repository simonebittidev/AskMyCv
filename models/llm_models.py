from typing import List
from pydantic import BaseModel, Field

class RewrittenQuestion(BaseModel):
    rewritten_question: str = Field(
        description="The clarified and rephrased version of the user's question, rewritten for maximum clarity based on conversation context."
    )