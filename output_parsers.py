from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class LegalDocumentIntel(BaseModel):
    summary:str = Field(description="Resumen corto del documento")
    generated_document:str = Field(description="Un documento parecido al ejemplo modificado al contexto")
    #facts:List[str]  = Field(description="Interesting facts about the person")
    

    def to_dict(self):
        return {
                "summary":self.summary,
                "generated_document": self.generated_document
            }
    

document_intel_parser = PydanticOutputParser(pydantic_object=LegalDocumentIntel)    