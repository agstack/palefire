"""
Pale Fire API Models

Pydantic models for API request/response validation.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class SearchMethod(str, Enum):
    """Available search methods."""
    standard = "standard"
    connection = "connection"
    question_aware = "question-aware"


class EpisodeType(str, Enum):
    """Episode content types."""
    text = "text"
    json = "json"


class KeywordExtractionMethod(str, Enum):
    """Available keyword extraction methods."""
    tfidf = "tfidf"
    textrank = "textrank"
    word_freq = "word_freq"
    combined = "combined"
    ner = "ner"


# ============================================================================
# Request Models
# ============================================================================

class Episode(BaseModel):
    """Episode for ingestion."""
    content: Any = Field(..., description="Episode content (string or object)")
    type: EpisodeType = Field(..., description="Content type")
    description: Optional[str] = Field(None, description="Episode description")


class IngestRequest(BaseModel):
    """Request to ingest episodes."""
    episodes: List[Episode] = Field(..., description="List of episodes to ingest")
    enable_ner: bool = Field(True, description="Enable NER enrichment")


class SearchRequest(BaseModel):
    """Request to search the knowledge graph."""
    query: str = Field(..., description="Search query", min_length=1)
    method: SearchMethod = Field(
        SearchMethod.question_aware,
        description="Search method to use"
    )
    limit: Optional[int] = Field(None, description="Maximum number of results", ge=1, le=100)


class KeywordExtractionRequest(BaseModel):
    """Request to extract keywords from text."""
    text: str = Field(..., description="Text to extract keywords from", min_length=1)
    method: KeywordExtractionMethod = Field(
        KeywordExtractionMethod.tfidf,
        description="Extraction method to use"
    )
    num_keywords: int = Field(10, description="Number of keywords to extract", ge=1, le=100)
    min_word_length: int = Field(3, description="Minimum word length", ge=1, le=50)
    max_word_length: int = Field(50, description="Maximum word length", ge=1, le=100)
    use_stemming: bool = Field(False, description="Enable stemming for preprocessing")
    tfidf_weight: float = Field(1.0, description="Weight for TF-IDF scores (combined method)", ge=0.0)
    textrank_weight: float = Field(0.5, description="Weight for TextRank scores (combined method)", ge=0.0)
    word_freq_weight: float = Field(0.3, description="Weight for word frequency scores (combined method)", ge=0.0)
    position_weight: float = Field(0.2, description="Weight for position-based scoring", ge=0.0)
    title_weight: float = Field(2.0, description="Weight multiplier for words in titles/headers", ge=0.0)
    first_sentence_weight: float = Field(1.5, description="Weight multiplier for words in first sentence", ge=0.0)
    enable_ngrams: bool = Field(True, description="Enable n-gram extraction (2-4 word phrases)")
    min_ngram: int = Field(2, description="Minimum n-gram size (1 for unigrams, 2-4 for phrases)", ge=1, le=4)
    max_ngram: int = Field(4, description="Maximum n-gram size (2, 3, or 4)", ge=2, le=4)
    ngram_weight: float = Field(1.2, description="Weight multiplier for n-grams", ge=0.0)
    documents: Optional[List[str]] = Field(None, description="Optional document corpus for IDF calculation")
    verify_ner: bool = Field(False, description="Verify NER results using LLM (only for method='ner')")
    deep: bool = Field(False, description="Process text sentence-by-sentence with ordered index (only for method='ner')")
    blocksize: int = Field(1, description="Number of sentences per block when deep=True (default: 1 = sentence-by-sentence)", ge=1)


# ============================================================================
# Response Models
# ============================================================================

class EntityInfo(BaseModel):
    """Entity information."""
    name: str
    type: Optional[str] = None
    labels: List[str] = []
    uuid: Optional[str] = None


class ConnectionInfo(BaseModel):
    """Connection information."""
    count: int
    entities: List[EntityInfo]
    relationship_types: List[str]


class ScoringInfo(BaseModel):
    """Scoring breakdown."""
    final_score: float
    original_score: float
    connection_score: Optional[float] = None
    temporal_score: Optional[float] = None
    query_match_score: Optional[float] = None
    entity_type_score: Optional[float] = None


class SearchResult(BaseModel):
    """Single search result."""
    rank: int
    uuid: str
    name: str
    summary: str
    labels: List[str]
    attributes: Dict[str, Any]
    scoring: Optional[ScoringInfo] = None
    connections: Optional[ConnectionInfo] = None
    recognized_entities: Optional[Dict[str, List[str]]] = None


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    method: str
    total_results: int
    results: List[SearchResult]
    timestamp: str


class StatusResponse(BaseModel):
    """Status response."""
    status: str
    message: str
    database_stats: Optional[Dict[str, int]] = None


class ConfigResponse(BaseModel):
    """Configuration response."""
    neo4j_uri: str
    llm_provider: str
    llm_model: str
    embedder_model: str
    search_method: str
    search_limit: int
    ner_enabled: bool


class KeywordInfo(BaseModel):
    """Single keyword information."""
    keyword: str
    score: float
    type: Optional[str] = Field(None, description="Entity type for NER, or 'unigram', '2-gram', '3-gram', '4-gram' for text-based extraction")
    reasoning: Optional[str] = Field(None, description="LLM reasoning for NER-based extraction")
    source: Optional[str] = Field(None, description="Source of entity: 'verified' (from spaCy), 'discovered' (new from LLM), 'corrected' (text variation from LLM)")


class KeywordExtractionResponse(BaseModel):
    """Keyword extraction response."""
    method: str
    num_keywords: int
    keywords: List[KeywordInfo]
    parameters: Dict[str, Any]
    verified: bool = Field(False, description="Whether the results were verified by LLM")
    timestamp: str


# ============================================================================
# File Parsing Models
# ============================================================================

class FileParseResponse(BaseModel):
    """File parsing response."""
    success: bool
    text: str = Field("", description="Extracted text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="File metadata")
    pages: List[str] = Field(default_factory=list, description="Page-by-page text (for multi-page documents)")
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted tables")
    keywords: Optional[List[KeywordInfo]] = Field(None, description="Extracted keywords from text")
    verified: bool = Field(False, description="Whether the keywords were verified by LLM")
    error: Optional[str] = Field(None, description="Error message if parsing failed")
    file_type: Optional[str] = Field(None, description="Detected file type")
    episodes: Optional[List[Dict[str, Any]]] = Field(None, description="Extracted episodes or structured vision data")
    timestamp: str


class ParseUrlRequest(BaseModel):
    """URL parsing request."""
    url: str = Field(..., description="URL to fetch and parse (web page or PDF)")
    timeout: int = Field(30, description="HTTP request timeout in seconds")
    extract_keywords: bool = Field(False, description="Extract keywords from content")
    keywords_method: str = Field("combined", description="Keyword extraction method")
    num_keywords: int = Field(20, description="Number of keywords to extract")
    vision_extract: bool = Field(False, description="Run Computer Vision extraction if the URL points to a PDF")
    vision_model: str = Field("gemma4:31b-cloud", description="Vision model for PDF image extraction")



# ============================================================================
# AI Agent Models
# ============================================================================

class AgentStatusResponse(BaseModel):
    """AI Agent daemon status response."""
    running: bool
    models_initialized: bool
    use_spacy: bool
    spacy_available: bool
    parsers_available: bool
    pid: Optional[int] = Field(None, description="Process ID if running")
    memory_mb: Optional[float] = Field(None, description="Memory usage in MB")
    cpu_percent: Optional[float] = Field(None, description="CPU usage percentage")
    timestamp: str


class EntityExtractionRequest(BaseModel):
    """Request to extract entities from text."""
    text: str = Field(..., description="Text to extract entities from", min_length=1)


class EntityExtractionResponse(BaseModel):
    """Entity extraction response."""
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="List of extracted entities")
    entities_by_type: Dict[str, List[str]] = Field(default_factory=dict, description="Entities grouped by type")
    all_entities: List[str] = Field(default_factory=list, description="All entity texts")
    timestamp: str


# ============================================================================
# Spatial / Geocoding Query Models
# ============================================================================

class SpatialQueryRequest(BaseModel):
    """Request to query a Vault dataset / document by place name or coordinates."""
    doc_id: str = Field(..., description="Vault document ID (e.g. 7FOuEOKGMJrGQFnbIdErKQ)")
    file_name: Optional[str] = Field(None, description="Specific report file (e.g. heatmap_report_page_10.csv or 7FOuEOKGMJrGQFnbIdErKQ_heatmap_report_page_10.csv)")
    page: Optional[int] = Field(None, description="Report page number if file_name not specified (e.g. 10)")
    location_name: Optional[str] = Field(None, description="Location to geocode via GeoNames / Nominatim (e.g. 'Sico', 'Tegucigalpa')")
    lat: Optional[float] = Field(None, description="Explicit latitude")
    lon: Optional[float] = Field(None, description="Explicit longitude")
    country_code: str = Field("HN", description="ISO country code (default 'HN')")
    geonames_username: Optional[str] = Field(None, description="Optional GeoNames username")


class SpatialQueryResponse(BaseModel):
    """Spatial query result containing geocoding information and matched data points."""
    doc_id: str
    file_name: str
    file_url: str
    resolved_location: Dict[str, Any] = Field(default_factory=dict, description="Resolved coordinates and geocoding source")
    matched_department: Optional[str] = Field(None, description="Department containing or closest to the coordinates")
    data_points: List[Dict[str, Any]] = Field(default_factory=list, description="Data points matching the location in the report")
    timestamp: str


class GeoIdQueryRequest(BaseModel):
    """Request to resolve coordinates into geolocation and optionally search a Vault document."""
    lat: Optional[float] = Field(None, description="Latitude (e.g. 13.34884)")
    lon: Optional[float] = Field(None, description="Longitude (e.g. -87.11769)")
    lng: Optional[float] = Field(None, description="Alias for longitude")
    coords: Optional[str] = Field(None, description="Coordinates string (e.g. '13.34884, -87.11769')")
    coordinates: Optional[str] = Field(None, description="Alias for coords string")
    location_name: Optional[str] = Field(None, description="Optional location / place name to resolve")
    docid: Optional[str] = Field(None, description="Optional Vault document ID or URL (e.g. NAfoXftXORhSUQZfalr2NQ)")
    doc_id: Optional[str] = Field(None, description="Alias for docid")


class GeoIdQueryResponse(BaseModel):
    """Response containing reverse-geocoded location, department, and matching document data points."""
    status: str = "success"
    coordinates: Dict[str, float] = Field(default_factory=dict, description="Resolved lat and lon")
    geolocation: Dict[str, Any] = Field(default_factory=dict, description="Detailed reverse-geocoded address/location info")
    matched_department: Optional[str] = Field(None, description="Department polygon matched via GeoPandas")
    doc_id: Optional[str] = Field(None, description="Vault document ID queried, if any")
    total_data_points: int = 0
    data_points: List[Dict[str, Any]] = Field(default_factory=list, description="Observations/predictions matching this location in the document")
    timestamp: str



