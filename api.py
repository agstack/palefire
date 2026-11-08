"""
Pale Fire FastAPI Wrapper

REST API for Pale Fire knowledge graph search system.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
from pathlib import Path

# Import Pale Fire components
import config
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from modules import EntityEnricher, QuestionTypeDetector, KeywordExtractor
from modules.api_models import (
    SearchMethod,
    EpisodeType,
    Episode,
    IngestRequest,
    SearchRequest,
    EntityInfo,
    ConnectionInfo,
    ScoringInfo,
    SearchResult,
    SearchResponse,
    StatusResponse,
    ConfigResponse,
    KeywordExtractionMethod,
    KeywordExtractionRequest,
    KeywordInfo,
    KeywordExtractionResponse,
    FileParseResponse,
    ParseUrlRequest,
    AgentStatusResponse,
    EntityExtractionRequest,
    EntityExtractionResponse,
    SpatialQueryRequest,
    SpatialQueryResponse,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pale Fire API",
    description="Intelligent Knowledge Graph Search System with 5-Factor Ranking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
graphiti_instance: Optional[Graphiti] = None
enricher_instance: Optional[EntityEnricher] = None
detector_instance: Optional[QuestionTypeDetector] = None
keyword_extractor_instance: Optional[KeywordExtractor] = None

# AI Agent daemon (optional)
try:
    from agents import get_daemon, AIAgentDaemon
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    logger.warning("AI Agent not available")


# ============================================================================
# Startup and Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize Graphiti on startup."""
    global graphiti_instance, enricher_instance, detector_instance, keyword_extractor_instance
    
    try:
        logger.info("Initializing Pale Fire API...")
        
        # Validate configuration
        config.validate_config()
        
        # Get configuration
        llm_cfg = config.get_llm_config()
        emb_cfg = config.get_embedder_config()
        
        # Configure LLM client
        llm_config = LLMConfig(
            api_key=llm_cfg['api_key'],
            model=llm_cfg['model'],
            small_model=llm_cfg['small_model'],
            base_url=llm_cfg['base_url'],
        )
        
        llm_client = OpenAIGenericClient(config=llm_config)
        
        # Initialize Graphiti
        graphiti_instance = Graphiti(
            uri=config.NEO4J_URI,
            user=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            llm_client=llm_client,
            embedder=OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=emb_cfg['api_key'],
                    embedding_model=emb_cfg['embedding_model'],
                    embedding_dim=emb_cfg['embedding_dim'],
                    base_url=emb_cfg['base_url'],
                )
            ),
            cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
        )
        
        # Initialize enricher and detector
        enricher_instance = EntityEnricher(use_spacy=config.NER_USE_SPACY)
        detector_instance = QuestionTypeDetector()
        
        # Initialize keyword extractor
        try:
            keyword_extractor_instance = KeywordExtractor()
            logger.info("✅ Keyword extractor initialized")
        except ImportError as e:
            logger.warning(f"Keyword extractor not available (gensim may not be installed): {e}")
            keyword_extractor_instance = None
        
        logger.info("✅ Pale Fire API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Pale Fire API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Close Graphiti on shutdown."""
    global graphiti_instance
    
    if graphiti_instance:
        try:
            await graphiti_instance.close()
            logger.info("✅ Pale Fire API shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint - API status."""
    return StatusResponse(
        status="ok",
        message="Pale Fire API is running"
    )


@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        async with graphiti_instance.driver.session() as session:
            result = await session.run("RETURN 1")
            await result.single()
        
        # Get database stats
        async with graphiti_instance.driver.session() as session:
            result = await session.run("""
                MATCH (n)
                RETURN count(n) as node_count,
                       count{(n)-[]->()} as relationship_count
            """)
            record = await result.single()
            stats = {
                "nodes": record['node_count'] if record else 0,
                "relationships": record['relationship_count'] if record else 0
            }
        
        return StatusResponse(
            status="healthy",
            message="All systems operational",
            database_stats=stats
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration."""
    return ConfigResponse(
        neo4j_uri=config.NEO4J_URI,
        llm_provider=config.LLM_PROVIDER,
        llm_model=config.OLLAMA_MODEL if config.LLM_PROVIDER == 'ollama' else config.OPENAI_MODEL,
        embedder_model=config.OLLAMA_EMBEDDING_MODEL if config.EMBEDDER_PROVIDER == 'ollama' else config.OPENAI_EMBEDDING_MODEL,
        search_method=config.DEFAULT_SEARCH_METHOD,
        search_limit=config.SEARCH_RESULT_LIMIT,
        ner_enabled=config.NER_ENABLED
    )


@app.post("/ingest", response_model=StatusResponse)
async def ingest_episodes(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest episodes into the knowledge graph.
    
    This endpoint accepts a list of episodes and ingests them into the graph.
    NER enrichment can be enabled/disabled per request.
    """
    try:
        logger.info(f"Ingesting {len(request.episodes)} episodes (NER: {request.enable_ner})")
        
        enricher = enricher_instance if request.enable_ner else None
        ingested_count = 0
        
        for i, episode in enumerate(request.episodes, 1):
            # Enrich episode if NER is enabled
            episode_data = {
                'content': episode.content,
                'type': episode.type.value,
                'description': episode.description or 'No description'
            }
            
            if enricher:
                episode_data = enricher.enrich_episode(episode_data)
            
            # Prepare content
            content = (episode_data['content'] if isinstance(episode_data['content'], str)
                      else str(episode_data['content']))
            
            # Add to Graphiti
            await graphiti_instance.add_episode(
                name=f"{config.EPISODE_NAME_PREFIX} {i}",
                episode_body=content,
                source=episode_data['type'],
                source_description=episode_data['description'],
                reference_time=datetime.now(timezone.utc),
            )
            
            ingested_count += 1
        
        return StatusResponse(
            status="success",
            message=f"Successfully ingested {ingested_count} episodes"
        )
        
    except Exception as e:
        logger.error(f"Error ingesting episodes: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search the knowledge graph.
    
    Performs intelligent search using the specified method and returns
    ranked results with detailed scoring information.
    """
    try:
        logger.info(f"Searching: '{request.query}' (method: {request.method.value})")
        
        # Use simplified search for API
        # For full functionality, extract search functions to a separate module
        from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
        
        # Execute basic search
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = request.limit or config.SEARCH_RESULT_LIMIT
        
        node_search_results = await graphiti_instance._search(
            query=request.query,
            config=node_search_config,
        )
        
        # Convert to dict format
        results = [{'node': node, 'original_score': getattr(node, 'score', 1.0)} 
                  for node in node_search_results.nodes]
        
        # Apply limit if specified
        limit = request.limit or config.SEARCH_TOP_K
        results = results[:limit]
        
        # Convert results to response format
        search_results = []
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                node = result.get('node')
                
                # Filter out name_embedding
                attributes = {}
                if hasattr(node, 'attributes') and node.attributes:
                    attributes = {k: v for k, v in node.attributes.items() if k != 'name_embedding'}
                
                # Build result
                search_result = SearchResult(
                    rank=i,
                    uuid=node.uuid if node else "",
                    name=node.name if node else "",
                    summary=node.summary if node else "",
                    labels=node.labels if hasattr(node, 'labels') else [],
                    attributes=attributes,
                )
                
                # Add scoring if available
                if 'final_score' in result:
                    search_result.scoring = ScoringInfo(
                        final_score=result.get('final_score', 0),
                        original_score=result.get('original_score', 0),
                        connection_score=result.get('connection_score'),
                        temporal_score=result.get('temporal_score'),
                        query_match_score=result.get('query_match_score'),
                        entity_type_score=result.get('entity_type_score'),
                    )
                
                # Add connections if available
                if 'connected_entities' in result:
                    entities = []
                    for entity in result['connected_entities']:
                        if isinstance(entity, dict):
                            entities.append(EntityInfo(
                                name=entity['name'],
                                type=entity.get('type'),
                                labels=entity.get('labels', []),
                                uuid=entity.get('uuid')
                            ))
                        else:
                            entities.append(EntityInfo(name=str(entity)))
                    
                    search_result.connections = ConnectionInfo(
                        count=result.get('connection_count', 0),
                        entities=entities,
                        relationship_types=result.get('relationship_types', [])
                    )
                
                # Add recognized entities if available
                if 'enriched_node' in result:
                    search_result.recognized_entities = result['enriched_node'].get('entities_by_type', {})
                
                search_results.append(search_result)
        
        return SearchResponse(
            query=request.query,
            method=request.method.value,
            total_results=len(search_results),
            results=search_results,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/keywords", response_model=KeywordExtractionResponse)
async def extract_keywords(request: KeywordExtractionRequest):
    """
    Extract keywords from text using Gensim.
    
    Supports multiple extraction methods (TF-IDF, TextRank, word frequency, combined)
    with configurable weights and parameters.
    """
    if keyword_extractor_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Keyword extraction not available. Install gensim: pip install gensim>=4.3.0"
        )
    
    try:
        logger.info(f"Extracting keywords (method: {request.method.value}, num: {request.num_keywords})")
        
        # If NER method, use AI Agent daemon
        if request.method.value == 'ner':
            if not AGENT_AVAILABLE:
                raise HTTPException(
                    status_code=503,
                    detail="NER keyword extraction requires AI Agent daemon."
                )
            
            daemon = get_daemon(use_spacy=True)
            if not daemon.model_manager.is_initialized():
                daemon.model_manager.initialize(use_spacy=True)
            
            keywords = daemon.extract_keywords(
                request.text,
                num_keywords=request.num_keywords,
                method='ner',
                verify_ner=request.verify_ner,
                deep=request.deep,
                blocksize=request.blocksize
            )
        else:
            # Create extractor with requested parameters
            extractor = KeywordExtractor(
                method=request.method.value,
                num_keywords=request.num_keywords,
                min_word_length=request.min_word_length,
                max_word_length=request.max_word_length,
                use_stemming=request.use_stemming,
                tfidf_weight=request.tfidf_weight,
                textrank_weight=request.textrank_weight,
                word_freq_weight=request.word_freq_weight,
                position_weight=request.position_weight,
                title_weight=request.title_weight,
                first_sentence_weight=request.first_sentence_weight,
                enable_ngrams=request.enable_ngrams,
                min_ngram=request.min_ngram,
                max_ngram=request.max_ngram,
                ngram_weight=request.ngram_weight,
            )
            
            # Extract keywords
            keywords = extractor.extract(request.text, request.documents)
        
        # Convert to response format
        keyword_list = [
            KeywordInfo(
                keyword=kw['keyword'],
                score=kw['score'],
                type=kw.get('type', 'unigram'),
                reasoning=kw.get('reasoning')
            )
            for kw in keywords
        ]
        
        # Build parameters dict
        parameters = {
            'num_keywords': request.num_keywords,
            'min_word_length': request.min_word_length,
            'max_word_length': request.max_word_length,
            'use_stemming': request.use_stemming,
            'tfidf_weight': request.tfidf_weight,
            'textrank_weight': request.textrank_weight,
            'word_freq_weight': request.word_freq_weight,
            'position_weight': request.position_weight,
            'title_weight': request.title_weight,
            'first_sentence_weight': request.first_sentence_weight,
            'enable_ngrams': request.enable_ngrams,
            'min_ngram': request.min_ngram,
            'max_ngram': request.max_ngram,
            'ngram_weight': request.ngram_weight,
            'verify_ner': request.verify_ner
        }
        
        return KeywordExtractionResponse(
            method=request.method.value,
            num_keywords=len(keyword_list),
            keywords=keyword_list,
            parameters=parameters,
            verified=request.verify_ner if request.method.value == 'ner' else False,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except ImportError as e:
        logger.error(f"Gensim not available: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Keyword extraction requires gensim: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        raise HTTPException(status_code=500, detail=f"Keyword extraction failed: {str(e)}")


@app.delete("/clean", response_model=StatusResponse)
async def clean_database():
    """
    Clean/clear the Neo4j database.
    
    ⚠️ WARNING: This permanently deletes all data!
    """
    try:
        logger.warning("Cleaning database...")
        
        # Get stats before cleaning
        async with graphiti_instance.driver.session() as session:
            result = await session.run("""
                MATCH (n)
                RETURN count(n) as node_count,
                       count{(n)-[]->()} as relationship_count
            """)
            record = await result.single()
            node_count = record['node_count'] if record else 0
            rel_count = record['relationship_count'] if record else 0
        
        if node_count == 0:
            return StatusResponse(
                status="success",
                message="Database is already empty"
            )
        
        # Delete all nodes and relationships
        async with graphiti_instance.driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
        
        logger.info(f"Database cleaned: {node_count} nodes, {rel_count} relationships deleted")
        
        return StatusResponse(
            status="success",
            message=f"Database cleaned successfully. Deleted {node_count} nodes and {rel_count} relationships."
        )
        
    except Exception as e:
        logger.error(f"Error cleaning database: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


# ============================================================================
# File Parsing Endpoints
# ============================================================================

@app.post("/parse", response_model=FileParseResponse)
async def parse_file(
    file: UploadFile = File(...),
    file_type: Optional[str] = Form(None, description="Override file type detection (txt, csv, pdf, xlsx, xls, ods)"),
    delimiter: Optional[str] = Form(None, description="CSV delimiter (for CSV files)"),
    include_headers: bool = Form(True, description="Include headers in CSV output"),
    max_pages: Optional[int] = Form(None, description="Maximum pages to parse (for PDF files)", ge=1),
    extract_tables: bool = Form(True, description="Extract tables (for PDF/spreadsheet files)"),
    sheet_names: Optional[str] = Form(None, description="Comma-separated sheet names (for spreadsheet files)"),
    extract_keywords: bool = Form(False, description="Extract keywords from parsed text"),
    keywords_method: str = Form("combined", description="Keyword extraction method (tfidf, textrank, word_freq, combined, ner)"),
    num_keywords: int = Form(20, description="Number of keywords to extract"),
    verify_ner: bool = Form(False, description="Verify NER results using LLM (only for keywords_method='ner')"),
    deep: bool = Form(False, description="Process text sentence-by-sentence with ordered index (only for keywords_method='ner')"),
    blocksize: int = Form(1, description="Number of sentences per block when deep=True (default: 1 = sentence-by-sentence)", ge=1),
):
    """
    Parse a file and extract text content.
    
    Supports multiple file types:
    - TXT: Plain text files
    - CSV: Comma-separated values
    - PDF: PDF documents
    - Spreadsheets: Excel (.xlsx, .xls) and OpenDocument (.ods)
    """
    if not AGENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="File parsing requires AI Agent. Install parsing dependencies: pip install PyPDF2 openpyxl xlrd odfpy"
        )
    
    try:
        # Save uploaded file to temporary location
        file_extension = Path(file.filename).suffix.lower() if file.filename else ''
        
        # Determine file type
        detected_type = file_type or file_extension.lstrip('.')
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Get daemon instance
            daemon = get_daemon(use_spacy=extract_keywords and keywords_method == 'ner')
            
            # Prepare parser options
            parser_options = {}
            if delimiter:
                parser_options['delimiter'] = delimiter
            if not include_headers:
                parser_options['include_headers'] = False
            if max_pages:
                parser_options['max_pages'] = max_pages
            if not extract_tables:
                parser_options['extract_tables'] = False
            if sheet_names:
                parser_options['sheet_names'] = [s.strip() for s in sheet_names.split(',')]
            
            # Parse file
            result = daemon.parse_file(tmp_path, file_type=detected_type, **parser_options)
            
            # Extract keywords if requested
            keywords = None
            if extract_keywords and result.get('success'):
                keywords_data = daemon.extract_keywords(
                    result.get('text', ''),
                    num_keywords=num_keywords,
                    method=keywords_method,
                    verify_ner=verify_ner,
                    deep=deep,
                    blocksize=blocksize
                )
                if keywords_data:
                    keywords = [
                        KeywordInfo(
                            keyword=kw['keyword'],
                            score=kw['score'],
                            type=kw.get('type'),
                            reasoning=kw.get('reasoning')
                        )
                        for kw in keywords_data
                    ]
            
            return FileParseResponse(
                success=result.get('success', False),
                text=result.get('text', ''),
                metadata=result.get('metadata', {}),
                pages=result.get('pages', []),
                tables=result.get('tables', []),
                keywords=keywords,
                verified=verify_ner if keywords_method == 'ner' else False,
                error=result.get('error'),
                file_type=detected_type,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        raise HTTPException(status_code=500, detail=f"File parsing failed: {str(e)}")


@app.post("/parse-url", response_model=FileParseResponse)
async def parse_url_endpoint(req: ParseUrlRequest):
    """
    Parse a web URL or download and process a PDF document.
    Supports Computer Vision extraction for maps and climatological data when vision_extract=True.
    """
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL must not be empty")

    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    is_pdf = parsed.path.lower().endswith(".pdf")

    # If it is a PDF or vision extraction requested
    if is_pdf:
        import tempfile
        import urllib.request
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            tmp_path = tmp_pdf.name

        try:
            logger.info(f"Downloading PDF from {url} to {tmp_path}")
            urllib.request.urlretrieve(url, tmp_path)

            episodes_data = None
            if req.vision_extract:
                try:
                    import subprocess
                    out_ep_path = tmp_path + ".episodes.json"
                    cv_timeout = max(req.timeout * 3, 1200)
                    cmd = [
                        "python3", "extract_computer_vision.py",
                        "--pdf", tmp_path,
                        "--vision-model", req.vision_model,
                        "--output", out_ep_path
                    ]
                    logger.info(f"Running CV extraction with timeout {cv_timeout}s: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True, timeout=cv_timeout)
                    if os.path.exists(out_ep_path):
                        with open(out_ep_path, "r", encoding="utf-8") as f:
                            episodes_data = json.load(f)
                        os.remove(out_ep_path)
                except Exception as cve:
                    logger.error(f"Computer Vision extraction failed: {cve}")

            daemon = get_daemon(use_spacy=req.extract_keywords and req.keywords_method == 'ner')
            result = daemon.parse_file(tmp_path, file_type="pdf")

            keywords = None
            if req.extract_keywords and result.get('success'):
                keywords_data = daemon.extract_keywords(
                    result.get('text', ''),
                    method=req.keywords_method,
                    num_keywords=req.num_keywords
                )
                if keywords_data:
                    keywords = [
                        KeywordInfo(
                            keyword=kw['keyword'],
                            score=kw['score'],
                            type=kw.get('type'),
                            reasoning=kw.get('reasoning')
                        )
                        for kw in keywords_data
                    ]

            return FileParseResponse(
                success=result.get('success', False),
                text=result.get('text', ''),
                metadata=result.get('metadata', {'url': url}),
                pages=result.get('pages', []),
                tables=result.get('tables', []),
                keywords=keywords,
                episodes=episodes_data,
                file_type="pdf",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        # Standard HTML URL parsing
        try:
            from agents.parsers import URLParser
            parser = URLParser(timeout=req.timeout)
            parse_res = parser.parse(url)
            res_dict = parse_res.to_dict()

            daemon = get_daemon(use_spacy=req.extract_keywords and req.keywords_method == 'ner')
            keywords = None
            if req.extract_keywords and res_dict.get('text'):
                keywords_data = daemon.extract_keywords(
                    res_dict.get('text', ''),
                    method=req.keywords_method,
                    num_keywords=req.num_keywords
                )
                if keywords_data:
                    keywords = [
                        KeywordInfo(
                            keyword=kw['keyword'],
                            score=kw['score'],
                            type=kw.get('type'),
                            reasoning=kw.get('reasoning')
                        )
                        for kw in keywords_data
                    ]

            return FileParseResponse(
                success=not bool(res_dict.get('error')),
                text=res_dict.get('text', ''),
                metadata=res_dict.get('metadata', {'url': url}),
                pages=res_dict.get('pages', []),
                tables=[],
                keywords=keywords,
                error=res_dict.get('error'),
                file_type="url",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        except Exception as e:
            logger.error(f"Error parsing URL {url}: {e}")
            raise HTTPException(status_code=500, detail=f"URL parsing failed: {str(e)}")



# ============================================================================
# AI Agent Endpoints
# ============================================================================

@app.get("/agent/status", response_model=AgentStatusResponse)
async def get_agent_status():
    """
    Get AI Agent daemon status.
    
    Returns information about the daemon's running state, model initialization,
    and system resources.
    """
    if not AGENT_AVAILABLE:
        return AgentStatusResponse(
            running=False,
            models_initialized=False,
            use_spacy=False,
            spacy_available=False,
            parsers_available=False,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    try:
        daemon = get_daemon(use_spacy=False)
        status = daemon.get_status()
        
        # Try to get process information if daemon is running
        pid = None
        memory_mb = None
        cpu_percent = None
        
        try:
            import psutil
            pidfile = '/tmp/palefire_ai_agent.pid'
            if os.path.exists(pidfile):
                with open(pidfile, 'r') as f:
                    pid = int(f.read().strip())
                
                try:
                    process = psutil.Process(pid)
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent(interval=0.1)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pid = None
        except ImportError:
            pass  # psutil not available
        
        return AgentStatusResponse(
            running=status.get('running', False),
            models_initialized=status.get('models_initialized', False),
            use_spacy=status.get('use_spacy', False),
            spacy_available=status.get('spacy_available', False),
            parsers_available=status.get('parsers_available', False),
            pid=pid,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")


@app.post("/entities", response_model=EntityExtractionResponse)
async def extract_entities(request: EntityExtractionRequest):
    """
    Extract named entities from text using spaCy or pattern-based extraction.
    
    Uses the AI Agent daemon if available for faster extraction (models stay loaded).
    Falls back to direct EntityEnricher if daemon is not available.
    """
    try:
        # Try to use daemon if available
        if AGENT_AVAILABLE:
            try:
                daemon = get_daemon(use_spacy=config.NER_USE_SPACY)
                
                # Initialize models if not already initialized
                if not daemon.model_manager.is_initialized():
                    daemon.model_manager.initialize(use_spacy=config.NER_USE_SPACY)
                
                # Extract entities using daemon
                entities_dict = daemon.extract_entities(request.text)
                
                return EntityExtractionResponse(
                    entities=entities_dict.get('entities', []),
                    entities_by_type=entities_dict.get('entities_by_type', {}),
                    all_entities=entities_dict.get('all_entities', []),
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            except Exception as e:
                logger.warning(f"Failed to use daemon for entity extraction: {e}, falling back to direct extraction")
        
        # Fall back to direct extraction
        if enricher_instance is None:
            raise HTTPException(
                status_code=503,
                detail="Entity extraction not available. EntityEnricher not initialized."
            )
        
        # Extract entities directly
        episode = {
            'content': request.text,
            'type': 'text',
            'description': 'Entity extraction'
        }
        
        enriched = enricher_instance.enrich_episode(episode)
        
        return EntityExtractionResponse(
            entities=enriched.get('entities', []),
            entities_by_type=enriched.get('entities_by_type', {}),
            all_entities=enriched.get('all_entities', []),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")


# ============================================================================
# Spatial & Geocoding Query Endpoints
# ============================================================================

def _geocode_place(query: str, country: str = "HN", geonames_user: Optional[str] = None):
    """Geocode a place name using GeoNames API with Nominatim fallback."""
    import urllib.request
    import urllib.parse
    import json

    # 1. Try GeoNames if username configured or provided
    user = geonames_user or os.environ.get("GEONAMES_USERNAME") or os.environ.get("GEONAMES_USER")
    if user:
        try:
            url = f"http://api.geonames.org/searchJSON?q={urllib.parse.quote(query)}&country={country}&maxRows=5&username={user}"
            req = urllib.request.Request(url, headers={"User-Agent": "PaleFire/1.0"})
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read().decode())
                if "geonames" in data and data["geonames"]:
                    g = data["geonames"][0]
                    return {
                        "source": "geonames",
                        "name": g.get("name"),
                        "lat": float(g.get("lat")),
                        "lon": float(g.get("lng")),
                        "adminName1": g.get("adminName1"),
                        "countryCode": g.get("countryCode")
                    }
        except Exception as e:
            logger.warning(f"GeoNames lookup failed: {e}")

    # 2. Fallback to OpenStreetMap Nominatim
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(query + ', Honduras')}&format=json&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "PaleFire/1.0 (info@codata.org)"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode())
            if data:
                item = data[0]
                return {
                    "source": "nominatim",
                    "name": item.get("display_name"),
                    "lat": float(item.get("lat")),
                    "lon": float(item.get("lon")),
                    "countryCode": country
                }
    except Exception as e:
        logger.warning(f"Nominatim lookup failed: {e}")

    return None


@app.post("/spatial-query", response_model=SpatialQueryResponse)
async def query_spatial_datapoints(request: SpatialQueryRequest):
    """
    Query Vault report data points by place name (via GeoNames / Nominatim) or coordinates.
    Finds the containing Honduran department via GeoPandas and retrieves data points from the target CSV.
    """
    import urllib.request
    import csv
    import io
    import re
    import json

    # 1. Resolve coordinates
    resolved_info = {}
    lat, lon = request.lat, request.lon

    if (lat is None or lon is None) and request.location_name:
        geo_res = _geocode_place(
            request.location_name,
            country=request.country_code,
            geonames_user=request.geonames_username
        )
        if not geo_res:
            raise HTTPException(
                status_code=404,
                detail=f"Could not geocode location '{request.location_name}' in country '{request.country_code}'"
            )
        resolved_info = geo_res
        lat = geo_res["lat"]
        lon = geo_res["lon"]
    elif lat is not None and lon is not None:
        resolved_info = {
            "source": "coordinates",
            "lat": lat,
            "lon": lon,
            "countryCode": request.country_code
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="Either location_name or (lat, lon) must be provided."
        )

    # 2. Match Honduran Department using GeoPandas
    matched_dept = None
    geojson_path = "datamaps/honduras_departments.geojson"
    if not os.path.exists(geojson_path):
        geojson_path = "/app/datamaps/honduras_departments.geojson"

    if os.path.exists(geojson_path):
        try:
            import geopandas as gpd
            from shapely.geometry import Point

            gdf = gpd.read_file(geojson_path)
            pt = Point(lon, lat)
            matching = gdf[gdf.geometry.contains(pt)]
            if not matching.empty:
                matched_dept = matching["shapeName"].values[0]
            else:
                # Nearest department boundary
                dists = gdf.geometry.distance(pt)
                matched_dept = gdf.loc[dists.idxmin(), "shapeName"]
        except Exception as e:
            logger.error(f"GeoPandas spatial matching error: {e}")
            # Fallback to adminName1 from geocoder if available
            matched_dept = resolved_info.get("adminName1")

    # 3. Resolve target file
    doc_id = request.doc_id
    if request.file_name:
        fname = request.file_name
        if not fname.startswith(doc_id):
            file_key = f"{doc_id}_{fname}"
        else:
            file_key = fname
    elif request.page is not None:
        file_key = f"{doc_id}_heatmap_report_page_{request.page:02d}.csv"
    else:
        file_key = f"{doc_id}_final_heatmap_report.csv"

    # 4. Fetch CSV content
    vault_base = os.environ.get("VAULT_URL", "https://ai.now.museum/vault")
    file_url = f"{vault_base}/{file_key}"

    data_rows = []
    try:
        # Check local MinIO or HTTP URL
        req = urllib.request.Request(file_url, headers={"User-Agent": "PaleFire/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            csv_text = r.read().decode("utf-8")

        # If final_heatmap_report, parse multi-row layout and attach legend and variable names
        raw_rows = list(csv.reader(io.StringIO(csv_text)))
        if len(raw_rows) >= 2 and raw_rows[1] and raw_rows[1][0] == "Legend":
            row1_legends = raw_rows[1]
            # Try to fetch croissant jsonld to get page descriptions and default report year
            page_desc = {}
            dataset_year = "2026"
            try:
                ld_url = f"{vault_base}/{doc_id}.jsonld"
                ld_req = urllib.request.Request(ld_url, headers={"User-Agent": "PaleFire/1.0"})
                with urllib.request.urlopen(ld_req, timeout=5) as r_ld:
                    ld_data = json.loads(r_ld.read().decode("utf-8"))
                    desc_text = str(ld_data.get("schema:description", "")) + " " + str(ld_data.get("description", ""))
                    m_yr = re.search(r"(?:19|20)\d{2}", desc_text)
                    if m_yr:
                        dataset_year = m_yr.group(0)
                    for dist in ld_data.get("distribution", []):
                        d_name = dist.get("name", "")
                        if "page_" in d_name and dist.get("encodingFormat") == "text/csv":
                            try:
                                p_idx = int(d_name.split("page_")[1].split(".")[0])
                                desc = dist.get("description", "")
                                if "(" in desc and ")" in desc:
                                    desc = desc.split("(", 1)[1].rsplit(")", 1)[0]
                                page_desc[p_idx] = desc
                            except Exception:
                                pass
            except Exception:
                pass

            dept_rows = [r for r in raw_rows[2:] if (matched_dept and r and matched_dept.lower() in r[0].lower()) or (not matched_dept and request.location_name and r and request.location_name.lower() in r[0].lower())]
            for idx, r in enumerate(dept_rows):
                if len(r) >= 5:
                    loc = r[0]
                    val = r[1]
                    date_val = r[2]
                    color = r[3]
                    anomaly = r[4]
                    leg = row1_legends[idx + 1] if idx + 1 < len(row1_legends) else ""
                    page_num = (idx // 2) + 1
                    side = "left" if idx % 2 == 0 else "right"
                    raw_desc = page_desc.get(page_num, "")
                    parts = [p.strip() for p in raw_desc.split("|")] if raw_desc else []
                    if side == "left":
                        panel_title = parts[0] if len(parts) > 0 and parts[0] else f"Report Page {page_num:02d}"
                    else:
                        panel_title = parts[1] if len(parts) > 1 and parts[1] else (parts[0] if len(parts) > 0 and parts[0] else f"Report Page {page_num:02d}")

                    # Year extraction: extract 4-digit years from title, date_val, or fallback to dataset_year
                    yrs = re.findall(r"(?:19|20)\d{2}", panel_title)
                    if not yrs:
                        date_yrs = re.findall(r"(?:19|20)\d{2}", date_val or "")
                        yrs = date_yrs if date_yrs else [dataset_year]
                    year_str = ", ".join(dict.fromkeys(yrs))

                    variable_name = f"{panel_title} ({side} panel)" if f"({side} panel)" not in panel_title.lower() else panel_title

                    data_rows.append({
                        "Location": loc,
                        "Page": page_num,
                        "Side": side,
                        "Year": year_str,
                        "Variable": variable_name,
                        "Value": val,
                        "Color": color,
                        "Anomaly": anomaly if anomaly != "Unknown" else val,
                        "Legend": leg
                    })
        else:
            reader = csv.DictReader(io.StringIO(csv_text))
            # Determine page number from request.page or file_key
            current_page = request.page
            if current_page is None:
                m_pg = re.search(r"page_(\d+)", file_key)
                if m_pg:
                    current_page = int(m_pg.group(1))

            for row in reader:
                loc = row.get("Location")
                if (matched_dept and loc and matched_dept.lower() in loc.lower()) or (not matched_dept and request.location_name and loc and request.location_name.lower() in loc.lower()):
                    var_str = row.get("Variable", "")
                    side = row.get("Side", "")
                    # Extract year from Variable or Date
                    yrs = re.findall(r"(?:19|20)\d{2}", var_str)
                    if not yrs:
                        yrs = re.findall(r"(?:19|20)\d{2}", row.get("Date", "") or "")
                    if not yrs:
                        yrs = ["2026"]
                    row["Year"] = ", ".join(dict.fromkeys(yrs))
                    if current_page is not None and "Page" not in row:
                        row["Page"] = current_page
                    if side and f"({side} panel)" not in var_str.lower():
                        row["Variable"] = f"{var_str} ({side} panel)"
                    data_rows.append(row)
    except Exception as e:
        logger.error(f"Failed to fetch or parse {file_url}: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Could not retrieve report data from {file_url}: {str(e)}"
        )

    return SpatialQueryResponse(
        doc_id=doc_id,
        file_name=file_key,
        file_url=file_url,
        resolved_location=resolved_info,
        matched_department=matched_dept,
        data_points=data_rows,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

