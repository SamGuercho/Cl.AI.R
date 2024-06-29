from typing import Optional, List, Any
import os

from genai.indexing.prompts import PromptType
from genai.indexing.prompt_builder import PromptBuilder
from openai import OpenAI
from llama_index.core.embeddings import BaseEmbedding

class NewsBot():
    """base class for news bot services"""
    _kb_name = os.environ['KB_NAME']
    def __init__(self, prompt_name:str|PromptType, model_name:str, embedding_model:BaseEmbedding, **kwargs):
        self._check_validity_prompt_name(prompt_name)
        self.prompt_builder = PromptBuilder(
            prompt_name=prompt_name,
            kb_name=self._kb_name,
            embedding_model=embedding_model)
        self.model_name = model_name
        self._client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    def _check_validity_prompt_name(self, prompt_name: str | PromptType) -> None:
        if isinstance(prompt_name, (str, PromptType)):
            if prompt_name in PromptType.__members__:
                return None
        if isinstance(prompt_name, str):
            if not prompt_name in PromptType.__members__:
                raise ValueError(f"Prompt name [{prompt_name}] not in the list of \
                valid PromptType: [{list(PromptType.__members__)}]")
        if not isinstance(prompt_name,  PromptType):
            raise ValueError(f"prompt_name must be of type str or PromptType, not {type(prompt_name)}")

    def create_completion(
            self,
            user_input: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            stop: Optional[List[str]] = None,
            **kwargs: Any
    ):
        instructions = self.prompt_builder.build_instruction_prompt(query=user_input)
        chat_prompt = self.prompt_builder.build_chat_prompt(
            system_prompt=instructions,
            user_input=user_input,
            **kwargs)

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=chat_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        )
        generated_text = response.choices[0].message.content
        return generated_text

