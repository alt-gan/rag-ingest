import warnings
warnings.filterwarnings("ignore")
import os
import json
import yaml
import argparse
import shutil
import datetime
from pathlib import Path
from tqdm.auto import tqdm
import logging
import traceback
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import qdrant_client
from extract import MarkdownPDFExtractor
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.together import TogetherLLM
from llama_index.llms.anthropic import Anthropic
from llama_index.core.llms import ChatMessage

import re
import torch
import nest_asyncio

torch.cuda.empty_cache()
nest_asyncio.apply()

# Load environment variables - TG_API_KEY, QDRANT_URL, QDRANT_API_KEY, ANTHROPIC_API_KEY
load_dotenv()

with open(Path('config/config.yaml').resolve(), 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

with open(Path('config/prompts.json').resolve(), 'r', encoding='utf-8') as f:
    prompts = json.load(f)

class Logger:
    @staticmethod
    def setup():
        log_dir = Path(__file__).parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{Path(__file__).stem}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

logger = Logger.setup()

class TextProcessor:
    @staticmethod
    def markdown_to_text(markdown_text):
        text = re.sub(r'```[\s\S]*?```', '', markdown_text)
        text = re.sub(r'`[^`\n]+`', '', text)
        text = re.sub(r'^#+\s+(.*?)$', r'\1', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[-*+]\s+(.*?)$', r'\1', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+(.*?)$', r'\1', text, flags=re.MULTILINE)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
        text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
        text = re.sub(r'\^([^\s^]+)(?:\^|(?=\s|$))', r'\1', text)
        text = re.sub(r'~([^\s~]+)(?:~|(?=\s|$))', r'\1', text)
        text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\|.*\|$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\+[-+]+\+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^>\s+(.*?)$', r'\1', text, flags=re.MULTILINE)
        text = re.sub(r'<[^>]+>', '', text)

        return text.strip()

class Index:
    # Class variables for models
    embed_model = None
    llm_model = None
    qdrant_client = None

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self.persist = os.path.join(Path(persist_dir).resolve(), collection_name)
        self.date = str(datetime.date.today())
        self.collection_name = collection_name

        self._initialize_components()

    @classmethod
    def _load_embed_model(cls):
        if cls.embed_model is None:
            cls.embed_model = HuggingFaceEmbedding(model_name=config['EMBED_MODEL'])
        return cls.embed_model
  
    @classmethod
    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=10, max=60))
    def _load_llm_model(cls):
        if cls.llm_model is None:
            cls.llm_model = Anthropic(
                model=config['ANTHROPIC_MODEL'],
                api_key=os.getenv('ANTHROPIC_API_KEY'),
                temperature=0.7,
                max_tokens=1024,
            )
        return cls.llm_model

    @classmethod
    def _get_qdrant_client(cls):
        if cls.qdrant_client is None:
            cls.qdrant_client = qdrant_client.QdrantClient(
                url=os.getenv('QDRANT_URL'),
                api_key=os.getenv('QDRANT_API_KEY')
            )
        return cls.qdrant_client

    def _initialize_components(self):
        self.splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=64)
        self.embed = self._load_embed_model()
        self.anthropic_llm = self._load_llm_model()
        
        Settings.llm = self.anthropic_llm
        Settings.embed_model = self.embed

        self.client = self._get_qdrant_client()
        self._setup_storage_context()

    def _setup_storage_context(self):
        if os.path.exists(self.persist) and ['docstore.json', 'index_store.json'] in os.listdir(self.persist):
            vector_store = QdrantVectorStore(client=self.client, collection_name=self.collection_name, enable_hybrid=True)
            self.storage_context = StorageContext.from_defaults(persist_dir=self.persist, vector_store=vector_store)
        else:
            self.client.delete_collection(collection_name=self.collection_name)
            vector_store = QdrantVectorStore(client=self.client, collection_name=self.collection_name, enable_hybrid=True)
            self.storage_context = StorageContext.from_defaults(vector_store=vector_store)


    def _process_chunk(self, chunk, highlighted_chunk, file_name, file_extension, category, idx, i):
        metadata = {
            "file_name": file_name,
            "extension": file_extension,
            "category": category,
            "created_at": self.date,
            "page_num": str(idx+1),
            "chunk_num": str(i+1),
            "highlighted_chunk": highlighted_chunk
        }

        return Document(
            id_=f"{file_name}_{str(idx+1)}_{str(i+1)}",
            text=chunk,
            metadata=metadata,
            excluded_llm_metadata_keys=["file_name", "created_at", "extension", "page_num", "chunk_num", "highlighted_chunk"],
            excluded_embed_metadata_keys=["file_name", "created_at", "extension", "page_num", "chunk_num", "highlighted_chunk"],
            metadata_seperator="\n",
            metadata_template="{key}: {value}",
            text_template="<METADATA>: {metadata_str}\n-----\n<CONTENT>: {content}"
        )

    def _process_page(self, args):
        idx, page, file_name, file_extension, category = args
        chunks = self.splitter.split_text(page)

        page_chunks = []
        for i, chunk in enumerate(chunks):
            highlighted_chunk = TextProcessor.markdown_to_text(chunk)
            page_chunks.append(self._process_chunk(chunk, highlighted_chunk, file_name, file_extension, category, idx, i))

        return page_chunks
    
    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=10, max=60))
    def _contextual_retrieval(self, document, chunk):
        messages = [
            ChatMessage(role="system", content=prompts["prompts"][0]["prompt_template"]),
            ChatMessage(
                role="user",
                content=[
                    {
                        "text": prompts["prompts"][1]["prompt_template"].format(
                            WHOLE_DOCUMENT=document
                        ),
                        "type": "text",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "text": prompts["prompts"][2]["prompt_template"].format(
                            CHUNK_CONTENT=chunk
                        ),
                        "type": "text",
                    },
                ],
            ),
        ]
        
        try:
            modified_chunk = str(self.anthropic_llm.chat(
                        messages,
                        extra_headers={"anthropic-beta": config['ANTHROPIC_PROMPT_CACHING']},
                    ))          
            return modified_chunk
        
        except Exception as err:
            logger.warning(f"Error during chunk modification: {str(err)}")
            raise


    def persist_docs(self, file_paths: list, category: str) -> None:
        all_chunks = []

        with tqdm(file_paths, desc="Processing files...", initial=1, total=len(file_paths), leave=False) as main_progress_bar:
            for index, file_path in enumerate(main_progress_bar, start=1):
                file_name = Path(file_path).name
                main_progress_bar.set_description(f"File {index}/{len(file_paths)}: {file_name}")

                file_extension = Path(file_path).suffix.lower()
                
                extractor = MarkdownPDFExtractor(file_path)
                markdown_pages = extractor.extract()

                with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                    futures = [executor.submit(self._process_page, (idx, page, file_name, file_extension, category)) for idx, page in enumerate(markdown_pages)]
                    
                    with tqdm(desc=f"Ingesting pages from {file_name}", initial=1, total=len(markdown_pages), leave=False) as progress_bar:
                        for future in as_completed(futures):
                            try:
                                result = future.result()
                                all_chunks.extend(result)
                            except Exception as e:
                                logger.error(f"Error processing page: {e}")
                                logger.exception(traceback.format_exc())
                            progress_bar.update(1)

                logger.info(f'Total No. of Chunks for {file_name}: {len(all_chunks)}')
        
        if self.client.collection_exists(collection_name=self.collection_name):
            index = load_index_from_storage(self.storage_context)
            index.refresh_ref_docs(all_chunks, update_kwargs={"delete_kwargs": {"delete_from_docstore": True}})
        else:
            shutil.rmtree(self.persist, ignore_errors=True)
            index = VectorStoreIndex.from_documents(documents=all_chunks, storage_context=self.storage_context)

        Path(self.persist).mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=self.persist)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input file or directory to Index", required=True)
    parser.add_argument("--file_category", help="File Category", required=True)
    parser.add_argument("--collection_name", default="rag_llm", help="Collection Name")
    parser.add_argument("--persist_dir", default="persist", help="Persistent Directory")

    args = parser.parse_args()
    
    input_path = Path(args.input).resolve()
    assert args.file_category in ["finance", "healthcare", "oil_gas"], "File category must be either `finance`, `healthcare`, or `oil_gas`"

    if input_path.is_file():
        assert input_path.suffix.lower() == '.pdf', "Input file must be a PDF"
        file_paths = [input_path]
    elif input_path.is_dir():
        file_paths = list(input_path.glob('*.pdf'))
        assert len(file_paths) > 0, "No PDF files found in the specified directory"
    else:
        raise ValueError("Invalid input: must be a PDF file or a directory containing PDF files")

    try:
        index_obj = Index(args.persist_dir, args.collection_name)
        index_obj.persist_docs(file_paths, args.file_category)
        logger.info(f"Indexing of {len(file_paths)} file(s) completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during indexing: {str(e)}")
        logger.exception(traceback.format_exc())

if __name__ == "__main__":
    main()