from typing import Any, List, Dict
from genai.indexing.prompts import PromptType
from genai.indexing.knowledge_base import KnowledgeBase, KBFactory
from llama_index.core.embeddings import BaseEmbedding
from dotenv import load_dotenv
import os
import warnings
from llama_index.core.schema import NodeWithScore, BaseNode

load_dotenv()
class PromptBuilder:
    _top_k_rag = int(os.environ['TOP_K_RAG'])
    _default_kb_type = os.environ['DEFAULT_KB_TYPE']
    def __init__(self, prompt_name: str | PromptType, kb_name:str, embedding_model:BaseEmbedding, kb_type:str|None=None, **kb_kwargs):
        self.prompt = self._get_prompt(prompt_name)
        self.kb = KBFactory.get_kb(
            kb_name=kb_name,
            kb_type=kb_type or self._default_kb_type,
            embedding_model=embedding_model
        )

    def _get_prompt(self, prompt_name:str|PromptType) -> str:
        return PromptType[prompt_name].value if isinstance(prompt_name, str) else prompt_name.value

    def build_instruction_prompt(self, query:str) -> str:
        """Builds the instruction prompt for the model."""
        # TODO Select the texts the most relevant to the query
        nodes = self.kb.search(query, top_k=self._top_k_rag)
        needed_labels = ['left', 'center-right', 'right', 'center', 'center-left']
        selected_nodes = []
        for node in nodes:
            if node.metadata.get('label') in needed_labels:
                selected_nodes.append(node)
                needed_labels.remove(node.metadata['label'])
            elif not node.metadata.get('label'):
                warnings.warn(f"Node {node.id_} has no label")
        unified_prompt = self.unify_nodes(selected_nodes)
        return self.prompt + "\n"+ "_______\n" + unified_prompt

    def unify_nodes(self, nodes:List[BaseNode]) -> str:
        unified_prompt = []
        for i, node in enumerate(nodes):
            i_prompt = f"Text {i}:\n"
            i_prompt += f"Label:{node.metadata['label']}\n"
            i_prompt += f"Text:{node.text}\n"
            unified_prompt.append(i_prompt)
        return "\n\n"+ "\n_______\n".join(unified_prompt)

    def build_chat_prompt(
            self,
            system_prompt:str,
            user_input: str,
            **kwargs: Any
    ) -> List[Dict[str, str]]:
        """Builds the chat prompt for the model."""
        # TODO build chat prompt here
        sys_message = self._prepare_message("system", system_prompt)
        usr_message = self._prepare_message("user", user_input)
        prompt_message = [sys_message, usr_message]

        return prompt_message

    def _prepare_message(self, role: str, content: str) -> Dict[str, str]:
        return {"role": role, "content": content}

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    prompt_builder = PromptBuilder("NEWSBOT_NEUTRALIZE", embedding_model)
    print(prompt_builder.build_instruction_prompt("What is the latest news on the US election?"))