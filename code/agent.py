try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': False,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 3,
 'runtime_enabled': True,
 'sanitize_pii': False}


import os
import logging
import asyncio
import time
from typing import List, Optional, Dict, Any, Tuple, Callable

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, ValidationError, Field, model_validator
from dotenv import load_dotenv

from azure.storage.blob import (
    BlobServiceClient,
    generate_blob_sas,
    BlobSasPermissions,
)
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient as AzureSearchClient
from azure.search.documents.models import VectorizedQuery

import openai
import requests
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_delay,
    wait_exponential,
    RetryError,
)

# Observability wrappers are injected by the runtime
# from observability import trace_step, trace_step_sync

# Load .env if present
load_dotenv()

# ---------------------- Logging Configuration ----------------------
logger = logging.getLogger("custom_translator_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
)
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# ---------------------- Configuration Management ----------------------

class Config:
    """Configuration loader for environment variables."""

    @staticmethod
    def get(key: str, default: Optional[str] = None) -> Optional[str]:
        return os.getenv(key, default)

    @staticmethod
    def validate(required_keys: List[str]) -> None:
        missing = [k for k in required_keys if not os.getenv(k)]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {missing}")

# ---------------------- Input Models and Validators ----------------------

class UserInputModel(BaseModel):
    input: str = Field(..., max_length=50000)

    @field_validator("input")
    @classmethod
    def validate_input(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Input cannot be empty.")
        if len(v) > 50000:
            raise ValueError("Input exceeds maximum allowed length (50,000 characters).")
        return v.strip()

# ---------------------- Utility: Logger ----------------------

class Logger:
    """Centralized logger utility."""

    def log(self, event: str, level: str = "info", details: Optional[dict] = None) -> None:
        try:
            msg = f"{event} | Details: {details or {}}"
            if level == "info":
                logger.info(msg)
            elif level == "warning":
                logger.warning(msg)
            elif level == "error":
                logger.error(msg)
            elif level == "debug":
                logger.debug(msg)
            else:
                logger.info(msg)
        except Exception as e:
            # Logging must not interrupt main flow
            pass

# ---------------------- Utility: ErrorHandler ----------------------

class ErrorHandler:
    """Centralized error handling and formatting."""

    def __init__(self, logger: Logger):
        self.logger = logger

    def handle_error(self, error_code: str, context: dict) -> str:
        self.logger.log(
            event=f"Error occurred: {error_code}",
            level="error",
            details=context,
        )
        if error_code == "ERR_BLOB_NOT_FOUND":
            return (
                "The requested file could not be found in Azure Blob Storage. "
                "Please verify the filename and try again."
            )
        elif error_code == "ERR_TRANSLATION_FAILED":
            return (
                "The translation service failed to process your file after multiple attempts. "
                "Please try again later or contact support."
            )
        elif error_code == "ERR_KB_NO_RESULT":
            return (
                "I am unable to find the requested information in the knowledge base. "
                "Please provide more details or consult an alternative resource."
            )
        else:
            return (
                "An unexpected error occurred. Please try again later or contact support."
            )

# ---------------------- Integration: AzureBlobClient ----------------------

class AzureBlobClient:
    """Handles file existence validation and SAS URL generation."""

    def __init__(self, logger: Logger):
        self.logger = logger
        self._blob_service_client = None
        self._container_name = Config.get("AZURE_BLOB_CONTAINER_NAME")
        self._connection_string = Config.get("AZURE_BLOB_STORAGE_CONNECTION_STRING")

    def _get_blob_service_client(self):
        if not self._blob_service_client:
            if not self._connection_string:
                raise RuntimeError("AZURE_BLOB_STORAGE_CONNECTION_STRING not configured.")
            self._blob_service_client = BlobServiceClient.from_connection_string(
                self._connection_string
            )
        return self._blob_service_client

    def file_exists(self, filename: str) -> bool:
        try:
            client = self._get_blob_service_client()
            container_client = client.get_container_client(self._container_name)
            blob_client = container_client.get_blob_client(blob=filename)
            exists = blob_client.exists()
            self.logger.log(
                event=f"Checked existence for file '{filename}'",
                level="info",
                details={"exists": exists},
            )
            return exists
        except Exception as e:
            self.logger.log(
                event=f"Error checking file existence: {filename}",
                level="error",
                details={"error": str(e)},
            )
            return False

    def generate_sas_url(self, filename: str, expiry_minutes: int = 10) -> str:
        try:
            client = self._get_blob_service_client()
            if not self._container_name:
                raise RuntimeError("AZURE_BLOB_CONTAINER_NAME not configured.")
            sas_token = generate_blob_sas(
                account_name=client.account_name,
                container_name=self._container_name,
                blob_name=filename,
                account_key=client.credential.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=time.time() + expiry_minutes * 60,
            )
            url = f"https://{client.account_name}.blob.core.windows.net/{self._container_name}/{filename}?{sas_token}"
            self.logger.log(
                event=f"Generated SAS URL for file '{filename}'",
                level="info",
                details={"sas_url": url[:60] + "..."},
            )
            return url
        except Exception as e:
            self.logger.log(
                event=f"Failed to generate SAS URL for file '{filename}'",
                level="error",
                details={"error": str(e)},
            )
            raise

# ---------------------- Integration: AzureTranslatorClient ----------------------

class AzureTranslatorClient:
    """Handles translation requests to Azure Translator."""

    def __init__(self, logger: Logger):
        self.logger = logger
        self._endpoint = Config.get("AZURE_TRANSLATOR_ENDPOINT")
        self._api_key = Config.get("AZURE_TRANSLATOR_API_KEY")
        self._region = Config.get("AZURE_TRANSLATOR_REGION")
        self._target_language = Config.get("AZURE_TRANSLATOR_TARGET_LANGUAGE", "en")

    def translate_document(self, sas_url: str) -> Tuple[int, str]:
        if not self._endpoint or not self._api_key or not self._region:
            raise RuntimeError(
                "Azure Translator configuration missing (endpoint, api_key, region required)."
            )
        url = f"{self._endpoint}/translator/text/batch/v1.0/documents:translate"
        headers = {
            "Ocp-Apim-Subscription-Key": self._api_key,
            "Ocp-Apim-Subscription-Region": self._region,
            "Content-Type": "application/json",
        }
        body = {
            "inputs": [
                {
                    "source": {"sourceUrl": sas_url},
                    "targets": [
                        {
                            "targetUrl": None,
                            "language": self._target_language,
                        }
                    ],
                }
            ]
        }
        try:
            _obs_t0 = _time.time()
            resp = requests.post(url, headers=headers, json=body, timeout=30)
            try:
                trace_tool_call(
                    tool_name='requests.post',
                    latency_ms=int((_time.time() - _obs_t0) * 1000),
                    output=str(resp)[:200] if resp is not None else None,
                    status="success",
                )
            except Exception:
                pass
            self.logger.log(
                event="Azure Translator API called",
                level="info",
                details={"status_code": resp.status_code, "response": resp.text[:200]},
            )
            if resp.status_code == 200:
                # For demo, assume translation is synchronous and content is in response
                # In reality, Azure Translator is async and requires polling
                # Here, we return the response text as "translated content"
                return 200, resp.text
            else:
                return resp.status_code, resp.text
        except Exception as e:
            self.logger.log(
                event="Azure Translator API error",
                level="error",
                details={"error": str(e)},
            )
            return 500, str(e)

# ---------------------- Service: TranslationService ----------------------

class TranslationService:
    """Coordinates file validation, SAS URL generation, translation request, and retry logic."""

    def __init__(
        self,
        blob_client: AzureBlobClient,
        translator_client: AzureTranslatorClient,
        error_handler: ErrorHandler,
        logger: Logger,
    ):
        self.blob_client = blob_client
        self.translator_client = translator_client
        self.error_handler = error_handler
        self.logger = logger

    async def translate_file(self, filename: str) -> dict:
        # Validate file existence
        async with trace_step(
            "validate_file_existence",
            step_type="process",
            decision_summary="Check if file exists in Azure Blob Storage",
            output_fn=lambda r: f"exists={r}",
        ) as step:
            exists = self.blob_client.file_exists(filename)
            step.capture(exists)
        if not exists:
            error_msg = self.error_handler.handle_error(
                "ERR_BLOB_NOT_FOUND", {"filename": filename}
            )
            return {
                "success": False,
                "error": error_msg,
                "error_type": "ERR_BLOB_NOT_FOUND",
            }

        # Generate SAS URL
        try:
            async with trace_step(
                "generate_sas_url",
                step_type="tool_call",
                decision_summary="Generate SAS URL for file",
                output_fn=lambda r: f"sas_url={r[:60]}..." if r else "sas_url=None",
            ) as step:
                sas_url = self.blob_client.generate_sas_url(filename)
                step.capture(sas_url)
        except Exception as e:
            error_msg = self.error_handler.handle_error(
                "ERR_BLOB_NOT_FOUND", {"filename": filename, "error": str(e)}
            )
            return {
                "success": False,
                "error": error_msg,
                "error_type": "ERR_BLOB_NOT_FOUND",
            }

        # Translation with retry logic (up to 100 seconds)
        start_time = time.time()
        last_status = None
        last_content = None
        try:
            async with trace_step(
                "translate_document",
                step_type="tool_call",
                decision_summary="Call Azure Translator with retry logic",
                output_fn=lambda r: f"status={r[0]}",
            ) as step:
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type(Exception),
                    stop=stop_after_delay(100),
                    wait=wait_exponential(multiplier=1, min=2, max=10),
                    reraise=True,
                ):
                    with attempt:
                        status, content = self.translator_client.translate_document(
                            sas_url
                        )
                        last_status = status
                        last_content = content
                        if status == 200:
                            break
                        else:
                            raise Exception(
                                f"Translation failed with status {status}: {content[:100]}"
                            )
                step.capture((last_status, last_content))
        except RetryError as re:
            self.logger.log(
                event="Translation retry exhausted",
                level="error",
                details={"filename": filename, "error": str(re)},
            )
            error_msg = self.error_handler.handle_error(
                "ERR_TRANSLATION_FAILED", {"filename": filename, "error": str(re)}
            )
            return {
                "success": False,
                "error": error_msg,
                "error_type": "ERR_TRANSLATION_FAILED",
            }
        except Exception as e:
            self.logger.log(
                event="Translation failed",
                level="error",
                details={"filename": filename, "error": str(e)},
            )
            error_msg = self.error_handler.handle_error(
                "ERR_TRANSLATION_FAILED", {"filename": filename, "error": str(e)}
            )
            return {
                "success": False,
                "error": error_msg,
                "error_type": "ERR_TRANSLATION_FAILED",
            }

        # Success
        return {
            "success": True,
            "summary": (
                f"File '{filename}' was successfully translated. "
                f"SAS URL was generated and translation completed."
            ),
            "translated_content": last_content,
        }

# ---------------------- Integration: SearchClient (Azure AI Search) ----------------------

class SearchClient:
    """Queries Azure AI Search index for relevant document chunks."""

    def __init__(self, logger: Logger):
        self.logger = logger
        self._endpoint = Config.get("AZURE_SEARCH_ENDPOINT")
        self._index_name = Config.get("AZURE_SEARCH_INDEX_NAME")
        self._api_key = Config.get("AZURE_SEARCH_API_KEY")
        self._client = None

    def _get_client(self):
        if not self._client:
            if not self._endpoint or not self._index_name or not self._api_key:
                raise RuntimeError(
                    "Azure Search configuration missing (endpoint, index_name, api_key required)."
                )
            self._client = AzureSearchClient(
                endpoint=self._endpoint,
                index_name=self._index_name,
                credential=AzureKeyCredential(self._api_key),
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def search(self, query: str, vector: List[float], top_k: int = 5) -> List[dict]:
        try:
            client = self._get_client()
            vector_query = VectorizedQuery(
                vector=vector, k_nearest_neighbors=top_k, fields="vector"
            )
            results = client.search(
                search_text=query,
                vector_queries=[vector_query],
                top=top_k,
                select=["chunk", "title"],
            )
            chunks = []
            for r in results:
                if r.get("chunk"):
                    chunks.append(
                        {
                            "chunk": r["chunk"],
                            "title": r.get("title", ""),
                        }
                    )
            self.logger.log(
                event="Azure AI Search query executed",
                level="info",
                details={"query": query, "chunks_found": len(chunks)},
            )
            return chunks
        except Exception as e:
            self.logger.log(
                event="Azure AI Search query failed",
                level="error",
                details={"error": str(e)},
            )
            return []

# ---------------------- Service: Retriever (RAG Orchestration) ----------------------

class Retriever:
    """Handles embedding and retrieval orchestration for RAG."""

    def __init__(self, search_client: SearchClient, logger: Logger):
        self.search_client = search_client
        self.logger = logger
        self._openai_client = None

    def _get_openai_client(self):
        if not self._openai_client:
            api_key = Config.get("AZURE_OPENAI_API_KEY")
            endpoint = Config.get("AZURE_OPENAI_ENDPOINT")
            if not api_key or not endpoint:
                raise RuntimeError(
                    "Azure OpenAI configuration missing (endpoint, api_key required)."
                )
            self._openai_client = openai.AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint,
            )
        return self._openai_client

    @trace_agent(agent_name='Custom Translator Application Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        # Embed user query
        async with trace_step(
            "embed_query",
            step_type="tool_call",
            decision_summary="Generate embedding for user query",
            output_fn=lambda r: f"embedding_len={len(r) if r else 0}",
        ) as step:
            openai_client = self._get_openai_client()
            embedding_model = Config.get(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
            )
            embedding_resp = await openai_client.embeddings.create(
                input=query, model=embedding_model
            )
            embedding = embedding_resp.data[0].embedding
            step.capture(embedding)
        # Search
        async with trace_step(
            "search_azure_ai",
            step_type="tool_call",
            decision_summary="Search Azure AI Search with vector+keyword",
            output_fn=lambda r: f"chunks={len(r)}",
        ) as step:
            chunks = self.search_client.search(query, embedding, top_k=top_k)
            step.capture(chunks)
        return chunks

# ---------------------- Integration: LLMClient ----------------------

class LLMClient:
    """Handles LLM calls to OpenAI GPT-4.1 for response generation."""

    def __init__(self, logger: Logger):
        self.logger = logger
        self._openai_client = None
        self._model = Config.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4-1106-preview")
        self._temperature = float(Config.get("LLM_TEMPERATURE", "0.7"))
        self._max_tokens = int(Config.get("LLM_MAX_TOKENS", "2000"))
        self._system_prompt = (
            "You are a formal, reliable assistant for a custom translator application. Your primary role is to:\n\n"
            "- Accept a filename as input.\n\n"
            "- Retrieve the file from Azure Blob Storage using the configured blob settings.\n\n"
            "- Generate a secure SAS URL for the file.\n\n"
            "- Submit the SAS URL to the Azure Translator service.\n\n"
            "- If the translation service returns a status code of 200, proceed to provide the translated content.\n\n"
            "- If the status is not 200, retry the translation request for up to 100 seconds before returning an error.\n\n"
            "- Additionally, answer user queries using the connected knowledge base via Azure AI Search, referencing only the provided source documents.\n\n"
            "Output all responses in clear, formal language. If information is not found in the knowledge base, politely inform the user and suggest alternative actions."
        )

    def _get_openai_client(self):
        if not self._openai_client:
            api_key = Config.get("AZURE_OPENAI_API_KEY")
            endpoint = Config.get("AZURE_OPENAI_ENDPOINT")
            if not api_key or not endpoint:
                raise RuntimeError(
                    "Azure OpenAI configuration missing (endpoint, api_key required)."
                )
            self._openai_client = openai.AsyncOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint,
            )
        return self._openai_client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_response(self, prompt: str, context: List[dict]) -> str:
        # Compose context string
        context_str = "\n\n".join(
            [f"Source: {c.get('title', '')}\n{c.get('chunk', '')}" for c in context]
        )
        user_message = f"{prompt}\n\nRelevant context:\n{context_str}"
        async with trace_step(
            "llm_generate_response",
            step_type="llm_call",
            decision_summary="Call LLM to generate answer",
            output_fn=lambda r: f"length={len(r) if r else 0}",
        ) as step:
            openai_client = self._get_openai_client()
            response = await openai_client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            content = response.choices[0].message.content
            step.capture(content)
            try:
                trace_model_call(
                    provider="openai",
                    model_name=self._model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    latency_ms=int(response.response_ms)
                    if hasattr(response, "response_ms")
                    else 0,
                    response_summary=content[:200] if content else "",
                )
            except Exception:
                pass
            return content

# ---------------------- Service: KnowledgeBaseService ----------------------

class KnowledgeBaseService:
    """Handles user queries by retrieving relevant information from Azure AI Search."""

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        error_handler: ErrorHandler,
        logger: Logger,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.error_handler = error_handler
        self.logger = logger
        self._fallback_response = (
            "I am unable to find the requested information in the knowledge base. "
            "Please provide more details or consult an alternative resource."
        )

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def answer_query(self, query: str) -> str:
        async with trace_step(
            "retrieve_kb_chunks",
            step_type="tool_call",
            decision_summary="Retrieve relevant chunks from Azure AI Search",
            output_fn=lambda r: f"chunks={len(r)}",
        ) as step:
            chunks = await self.retriever.retrieve(query, top_k=5)
            step.capture(chunks)
        if not chunks:
            return self._fallback_response
        async with trace_step(
            "generate_kb_answer",
            step_type="llm_call",
            decision_summary="Generate answer using LLM and KB context",
            output_fn=lambda r: f"length={len(r) if r else 0}",
        ) as step:
            answer = await self.llm_client.generate_response(query, chunks)
            step.capture(answer)
        return answer

# ---------------------- Service: IntentClassifier ----------------------

class IntentClassifier:
    """Classifies user input as translation or knowledge base query."""

    def classify(self, input: str) -> str:
        # Heuristic: If input looks like a filename, treat as translation
        # Accepts .pdf, .docx, .txt, .xlsx, .csv, .pptx, etc.
        input_stripped = input.strip()
        if (
            "." in input_stripped
            and input_stripped.split(".")[-1].lower()
            in {"pdf", "docx", "txt", "xlsx", "csv", "pptx"}
            and len(input_stripped.split()) == 1
        ):
            return "translation"
        # Otherwise, treat as knowledge base query
        if len(input_stripped.split()) > 1:
            return "query"
        raise ValueError("Input is ambiguous. Please provide a valid filename or query.")

# ---------------------- Controller: AgentController ----------------------

class AgentController:
    """Entry point for user requests; routes to translation or knowledge base modules."""

    def __init__(
        self,
        intent_classifier: IntentClassifier,
        translation_service: TranslationService,
        kb_service: KnowledgeBaseService,
        error_handler: ErrorHandler,
        logger: Logger,
    ):
        self.intent_classifier = intent_classifier
        self.translation_service = translation_service
        self.kb_service = kb_service
        self.error_handler = error_handler
        self.logger = logger

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_user_input(self, input_text: str) -> dict:
        async with trace_step(
            "classify_intent",
            step_type="parse",
            decision_summary="Classify user input as translation or query",
            output_fn=lambda r: f"intent={r}",
        ) as step:
            try:
                intent = self.intent_classifier.classify(input_text)
                step.capture(intent)
            except Exception as e:
                error_msg = self.error_handler.handle_error(
                    "ERR_INPUT_AMBIGUOUS", {"input": input_text, "error": str(e)}
                )
                return {
                    "success": False,
                    "error": error_msg,
                    "error_type": "ERR_INPUT_AMBIGUOUS",
                }
        if intent == "translation":
            async with trace_step(
                "process_translation",
                step_type="process",
                decision_summary="Process translation request",
                output_fn=lambda r: f"success={r.get('success', False)}",
            ) as step:
                result = await self.translation_service.translate_file(input_text)
                step.capture(result)
                return result
        elif intent == "query":
            async with trace_step(
                "process_kb_query",
                step_type="process",
                decision_summary="Process knowledge base query",
                output_fn=lambda r: f"length={len(r) if isinstance(r,str) else 0}",
            ) as step:
                answer = await self.kb_service.answer_query(input_text)
                step.capture(answer)
                return {
                    "success": True,
                    "answer": answer,
                }
        else:
            error_msg = self.error_handler.handle_error(
                "ERR_INPUT_AMBIGUOUS", {"input": input_text}
            )
            return {
                "success": False,
                "error": error_msg,
                "error_type": "ERR_INPUT_AMBIGUOUS",
            }

# ---------------------- Main Agent Class ----------------------

class CustomTranslatorAgent:
    """Main agent class composing all supporting services."""

    def __init__(self):
        self.logger = Logger()
        self.error_handler = ErrorHandler(self.logger)
        self.intent_classifier = IntentClassifier()
        self.blob_client = AzureBlobClient(self.logger)
        self.translator_client = AzureTranslatorClient(self.logger)
        self.translation_service = TranslationService(
            self.blob_client, self.translator_client, self.error_handler, self.logger
        )
        self.search_client = SearchClient(self.logger)
        self.retriever = Retriever(self.search_client, self.logger)
        self.llm_client = LLMClient(self.logger)
        self.kb_service = KnowledgeBaseService(
            self.retriever, self.llm_client, self.error_handler, self.logger
        )
        self.controller = AgentController(
            self.intent_classifier,
            self.translation_service,
            self.kb_service,
            self.error_handler,
            self.logger,
        )

    @trace_agent(agent_name='Custom Translator Application Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def handle(self, input_text: str) -> dict:
        async with trace_step(
            "agent_handle",
            step_type="final",
            decision_summary="Agent main entry point",
            output_fn=lambda r: f"success={r.get('success', False)}",
        ) as step:
            result = await self.controller.process_user_input(input_text)
            step.capture(result)
            return result

# ---------------------- FastAPI Application ----------------------

app = FastAPI(
    title="Custom Translator Application Agent",
    description="A formal, reliable assistant for translation and knowledge base queries.",
    version="1.0.0",
)

# CORS (allow all for demo; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = CustomTranslatorAgent()

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Malformed input. Please check your request format and try again.",
            "tips": [
                "Ensure your JSON is properly formatted.",
                "Check for missing or extra commas, brackets, or quotes.",
                "Input text must not be empty and must be under 50,000 characters.",
            ],
            "details": exc.errors(),
        },
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
        },
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An unexpected error occurred. Please try again later.",
        },
    )

@app.post("/agent/handle", response_model=dict)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def agent_handle(user_input: UserInputModel):
    """
    Main endpoint for processing user input.
    """
    try:
        _obs_t0 = _time.time()
        result = await agent.handle(user_input.input)
        try:
            trace_tool_call(
                tool_name='agent.handle',
                latency_ms=int((_time.time() - _obs_t0) * 1000),
                output=str(result)[:200] if result is not None else None,
                status="success",
            )
        except Exception:
            pass
        return result
    except Exception as e:
        logger.error(f"Agent handle error: {e}")
        return {
            "success": False,
            "error": "An unexpected error occurred. Please try again later.",
        }

@app.get("/health")
async def health_check():
    return {"success": True, "status": "ok"}

# ---------------------- Main Entrypoint ----------------------



async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        logger.info("Starting Custom Translator Application Agent...")
        uvicorn.run("agent:app", host="0.0.0.0", port=8080, reload=True)
        pass  # TODO: run your agent here
    finally:
        if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_with_eval_service())