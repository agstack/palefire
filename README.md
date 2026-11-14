# Pale Fire - Intelligent Knowledge Graph Search System
This framework is being developed by [Slava Tykhonov](https://www.linkedin.com/in/vyacheslavtikhonov/) and is highly experimental.

**Pale Fire is hosted by [AgStack](https://agstack.org/) of the [Linux Foundation](https://www.linuxfoundation.org/).**

> Named after Vladimir Nabokov's novel ["Pale Fire"](https://en.wikipedia.org/wiki/Pale_Fire), where a poem becomes the subject of extensive commentary and interpretation—just like how this system builds a rich knowledge graph from text and enables intelligent exploration through questions. 

"The novel is presented as a 999-line poem, written by the fictional poet John Shade, with a foreword, lengthy commentary, and index written by Shade's neighbor and academic colleague, Charles Kinbote. Together these elements form a narrative in which both fictional authors are central characters. Pale Fire's unusual structure has attracted much attention, and it is often cited as an important example of metafiction, as well as an analog precursor to hypertext fiction, and a poioumenon."

## Table of Contents

- [Overview](#overview)
- [Example](#example)
- [Quick Start](#quick-start)
- [Inference Engine](#inference-engine)
- [CLI Usage](#cli-usage)
- [API Usage](#api-usage)
- [Features](#features)
- [CLI Commands](#cli-commands)
- [Episode File Format](#episode-file-format)
- [Architecture](#architecture)
- [5-Factor Ranking System](#5-factor-ranking-system)
- [Question Types](#question-types)
- [Entity Types](#entity-types)
- [Configuration](#configuration)
- [Examples](#examples)
- [Documentation](#documentation)
- [Testing](#testing)
- [Requirements](#requirements)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [AI Agent Daemon](#ai-agent-daemon)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## Example

Pale Fire can transform factually correct and evidence-confirmed data points from research datasets into human-readable descriptions based on annotations created by querying knowledge graphs on event entities and LLM integration of new knowledge into understandable narratives. In opposite, it can turn back any human-readable annotation to factually correct data points and link the provenance information as reference.

**Use Case Example**: If you have data observations on strikes registered on a specific date and place, Pale Fire can:

- Query the knowledge graph for related entities (location, date, event type)
- Retrieve contextual information such as weather conditions and temperature from connected nodes
- Find evidence from witnesses and related sources
- Synthesize all this information into a coherent, human-readable narrative

For instance, given a data point like:
```
Event: Strike
Date: March 15, 2023
Location: San Francisco, CA
```

Pale Fire can generate a narrative that includes:
- Historical weather data for that date and location
- Temperature records and conditions
- Related witness accounts or news reports
- Contextual information about similar events
- Temporal relationships to other events in the knowledge graph

**Example Narrative Output**:

> On March 15, 2023, a labor strike occurred in San Francisco, California. The day was characterized by mild spring weather, with temperatures reaching 62°F (17°C) and partly cloudy conditions—typical for early spring in the Bay Area. According to weather records, the morning began with light fog that cleared by mid-day, providing clear visibility for the demonstration that took place in the city's financial district.
>
> Witness accounts from local news reports indicate that approximately 500 workers gathered outside the headquarters of a major tech company, carrying signs and chanting demands for better working conditions. The strike was part of a broader wave of labor actions that had been occurring across California's tech sector throughout early 2023, following similar events in Los Angeles on March 8th and Oakland on March 12th.
>
> This event was temporally connected to a series of related labor actions: it occurred just one week after a similar strike in Seattle, Washington, and preceded another major demonstration in San Jose scheduled for March 22nd. The knowledge graph reveals that these events were part of a coordinated effort by tech workers' unions across the West Coast, responding to industry-wide concerns about workplace safety and compensation.
>
> Historical context from the knowledge graph shows that San Francisco has a long history of labor activism, with notable strikes occurring in 2018 and 2020. The 2023 strike shares similar characteristics with these previous events, particularly in terms of location (financial district) and participant demographics (tech sector workers).

This transforms raw data points into rich, contextualized stories that are both factually accurate and humanly comprehensible.

## Overview

Pale Fire is an advanced knowledge graph search system featuring:

- **🧠 Question-Type Detection** - Automatically understands WHO/WHERE/WHEN/WHAT/WHY/HOW questions
- **🏷️ NER Enrichment** - Extracts and tags 18+ entity types (PER, LOC, ORG, DATE, etc.)
- **📊 5-Factor Ranking** - Combines semantic, connectivity, temporal, query matching, and entity-type intelligence
- **⚡ CLI Interface** - Easy-to-use command-line interface for ingestion and queries
- **🔧 Modular Architecture** - Clean separation of concerns for maintainability
- **🤖 AI Agent Daemon** - Long-running daemon service that keeps Gensim and spaCy models loaded in memory for instant access
- **🔑 Keyword Extraction** - Extract keywords and n-grams (2-4 words) using Gensim with configurable weights (TF-IDF, TextRank, Word Frequency)
- **📄 File Parsing** - Extract text from multiple formats: TXT, CSV, PDF, Excel (.xlsx, .xls), OpenDocument (.ods), URLs/HTML
- **👻 Ghostwriter** - RAG-based Q&A system that ingests content from URLs and answers questions using LLMs
- **📚 Theoretical Foundation** - Based on Pale Fire's interpretive framework (see [docs/PROS-CONS.md](docs/PROS-CONS.md))

## Quick Start

### Docker (Recommended)

```bash
# 1. Start all services
docker-compose up -d

# 2. Setup (pull models)
make setup

# 3. Ingest demo data
make ingest-demo

# 4. Run a query
make query

# 5. Access services
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Neo4j: http://localhost:7474
```

See **[docs/DOCKER.md](docs/DOCKER.md)** for complete Docker documentation.

## Inference Engine

Pale Fire supports multiple inference providers for intelligent exploration and data extraction.

### Ollama (Local)

Ollama is used as the default LLM provider for local inference and RAG capabilities.

**Installation**:
- **macOS**: `brew install ollama` or download from [ollama.com](https://ollama.com/download/mac)
- **Windows**: Download from [ollama.com](https://ollama.com/download/windows)
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`

**Pull Required Models**:
```bash
# Default model for inference
ollama pull gemma3:27b

# Alternative models
ollama pull deepseek-r1:7b
```

### Google Gemini (AI Footnotes)

The Gemini CLI provides cloud-based inference with support for agent skills and session restoration.

**Installation**:
Requires [Node.js](https://nodejs.org/) to be installed.
```bash
npm install -g gemini-chat-cli
```

**Configuration**:
1. Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
2. Set the `GOOGLE_API_KEY` environment variable:
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```

**Browser Extension Interop**: 
The backend automatically detects the `gemini` binary in your PATH or at common locations like `/opt/homebrew/bin/gemini`.

## CLI Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Install keyword extraction (optional but recommended)
pip install gensim>=4.3.0
# Optional: For better stemming support
pip install nltk

# 2. Configure environment
cp env.example .env  # Edit with your settings

# 3. View configuration
python palefire-cli.py config

# 4. Ingest demo data
python palefire-cli.py ingest --demo

# 5. Ask a question
python palefire-cli.py query "Who was the California Attorney General in 2020?"
```

## API Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp env.example .env  # Edit with your settings

# 3. Start API server
python api.py

# 4. Access API
# - Base URL: http://localhost:8000
# - Interactive docs: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
```

### GeoID & Reverse Geocoding API (`/api/geoid`)

The `/api/geoid` endpoint turns geographical coordinates into reverse-geocoded addresses, matches administrative boundaries (e.g., Honduran department polygons via GeoPandas), and can optionally cross-reference matching observations, predictions, and climate legends from a Vault document (`docid`).

#### Query Parameters:
- `lat` *(float)* & `lon` / `lng` *(float)*: Latitude and longitude.
- `coords` or `coordinates` *(string)*: Comma- or space-separated coordinates string (e.g. `"13.34884, -87.11769"`).
- `docid` or `doc_id` *(string, optional)*: Vault document ID or URL (e.g. `NAfoXftXORhSUQZfalr2NQ` or `https://ai.mediaquantum.eu/vault/doc/NAfoXftXORhSUQZfalr2NQ`).
- `location_name` *(string, optional)*: Place name fallback (e.g. `"Choluteca"`).

#### Examples:

**1. Reverse Geocode Coordinates to Location & Department:**

```bash
curl -s "http://localhost:8000/api/geoid?coords=13.34884,-87.11769"
```

*Response:*
```json
{
  "status": "success",
  "coordinates": {
    "lat": 13.34884,
    "lon": -87.11769
  },
  "geolocation": {
    "display_name": "Residencial Villas Madrigal, Choluteca, Honduras",
    "address": {
      "residential": "Residencial Villas Madrigal",
      "city": "Choluteca",
      "state": "Choluteca",
      "country": "Honduras",
      "country_code": "hn"
    }
  },
  "matched_department": "Choluteca",
  "doc_id": null,
  "total_data_points": 0,
  "data_points": []
}
```

**2. Query Location Data Points from a Vault Document:**

```bash
curl -s "http://localhost:8000/api/geoid?coords=13.34884,-87.11769&docid=https://ai.mediaquantum.eu/vault/doc/NAfoXftXORhSUQZfalr2NQ"
```

*Response (excerpt):*
```json
{
  "status": "success",
  "coordinates": {
    "lat": 13.34884,
    "lon": -87.11769
  },
  "matched_department": "Choluteca",
  "doc_id": "NAfoXftXORhSUQZfalr2NQ",
  "total_data_points": 56,
  "data_points": [
    {
      "@type": "Observation",
      "name": "Choluteca",
      "latitude": 13.34884,
      "longitude": -87.11769,
      "color": "Yellow-Green",
      "hex": "#b5db8a",
      "palette_hex": "#b1f57e",
      "anomaly": "25.1 - 50 mm",
      "value_range": "25.1 - 50 mm",
      "page": 10,
      "side": "left",
      "variable_page_title": "REGISTRO DE LLUVIA ACUMULADA GOBIERNO on 6} cENAos (Page 10)"
    }
  ]
}
```

**3. JSON POST Request:**

```bash
curl -s -X POST "http://localhost:8000/api/geoid" \
  -H "Content-Type: application/json" \
  -d '{
    "coords": "13.34884, -87.11769",
    "docid": "NAfoXftXORhSUQZfalr2NQ"
  }'
```


## Features

### Intelligent Question Detection

Automatically detects 8 question types and adjusts entity weights:

```bash
# WHO questions → boost person entities 2.0x
python palefire-cli.py query "Who was the Attorney General?"

# WHERE questions → boost location entities 2.0x
python palefire-cli.py query "Where did Kamala Harris work?"

# WHEN questions → boost date entities 2.0x
python palefire-cli.py query "When did Gavin Newsom become governor?"
```

### NER-Enriched Ingestion

Extract entities automatically during ingestion:

```bash
# With NER enrichment (recommended)
python palefire-cli.py ingest --file episodes.json

# Without NER (faster)
python palefire-cli.py ingest --file episodes.json --no-ner
```

### Multiple Search Methods

Choose the best method for your query:

```bash
# Question-aware (recommended for natural questions)
python palefire-cli.py query "Who is Gavin Newsom?" -m question-aware

# Connection-based (for finding central entities)
python palefire-cli.py query "Important people" -m connection

# Standard (fastest, basic RRF)
python palefire-cli.py query "California" -m standard
```

## CLI Commands

### Ingest Episodes

```bash
# From file
python palefire-cli.py ingest --file episodes.json

# Demo data
python palefire-cli.py ingest --demo

# Without NER
python palefire-cli.py ingest --file episodes.json --no-ner
```

### Query Knowledge Graph

```bash
# Basic query
python palefire-cli.py query "Your question here?"

# With specific method
python palefire-cli.py query "Your question?" --method question-aware

# Export results to JSON
python palefire-cli.py query "Your question?" --export results.json

# Combine method and export
python palefire-cli.py query "Who is X?" -m standard -e output.json

# Short form
python palefire-cli.py query "Who is X?" -m standard
```

### Show Configuration

```bash
python palefire-cli.py config
```

### Clean Database

```bash
# Clean database (with confirmation prompt)
python palefire-cli.py clean

# Clean without confirmation
python palefire-cli.py clean --confirm

# Delete only nodes (keep database structure)
python palefire-cli.py clean --nodes-only
```

### Extract Keywords

```bash
# Extract keywords from text
python palefire-cli.py keywords "Your text here" --num-keywords 10

# With n-grams (2-4 word phrases)
python palefire-cli.py keywords "Your text here" --min-ngram 2 --max-ngram 3

# Using specific method (tfidf, textrank, frequency, combined)
python palefire-cli.py keywords "Your text" --method combined

# Save to file
python palefire-cli.py keywords "Your text" -o results.json
```

### Parse Files

```bash
# Auto-detect file type and parse
python palefire-cli.py parse document.pdf

# Parse specific file types
python palefire-cli.py parse-txt document.txt
python palefire-cli.py parse-csv data.csv
python palefire-cli.py parse-pdf document.pdf
python palefire-cli.py parse-spreadsheet data.xlsx
python palefire-cli.py parse-url https://example.com

# Parse with options
python palefire-cli.py parse-csv data.csv --delimiter ";"
python palefire-cli.py parse-pdf document.pdf --max-pages 10
python palefire-cli.py parse-url https://example.com --extract-keywords --keywords-method ner
```

### Manage AI Agent Daemon

```bash
### AI Agent Daemon
*   **Start**: `python palefire-cli.py agent start --daemon`
*   **Status**: `python palefire-cli.py agent status`
*   **Stop**: `python palefire-cli.py agent stop`

# Restart daemon
python palefire-cli.py agent restart --daemon
```

### Get Help

```bash
python palefire-cli.py --help
python palefire-cli.py ingest --help
python palefire-cli.py query --help
python palefire-cli.py keywords --help
python palefire-cli.py parse --help
python palefire-cli.py agent --help
```

### Ghostwriter (RAG & Web Ingestion)

Ghostwriter allows you to ingest content from URLs and ask questions based on that knowledge.

```bash
# 1. Initialize configuration (if not already done)
python palefire-cli.py init

# 2. Ingest a URL
python palefire-cli.py ghostwriter ingest "https://example.com/article" --collection my-knowledge

# 3. Ask a question
python palefire-cli.py ghostwriter ask "What is the article about?" --collection my-knowledge

# 4. Search the knowledge base
python palefire-cli.py ghostwriter search "specific keyword" --collection my-knowledge
```

### Computer Vision Map Extraction

Pale Fire includes a powerful computer vision module that can interpret heatmaps and climatological data from images and PDFs, extract spatial longitude/latitude coordinates, read the color scales, and generate a final CSV report using the LLM and the `datamaps` spatial tools.

```bash
# Ensure your Ollama host is set correctly
export OLLAMA_HOST=http://localhost:11434

# 1. Process a single map image
python extract_computer_vision.py --image path/to/map.jpg

# 2. Process a directory of images (e.g., extracted pages)
python extract_computer_vision.py --image-dir datamaps/data/pdf-extraction/

# 3. Download and process a PDF directly from a URL
python extract_computer_vision.py --pdf-url "https://example.com/report.pdf"

# 4. Process a local PDF directly
python extract_computer_vision.py --pdf path/to/report.pdf

# Note: The system will automatically detect multi-panel maps (like 2x2 grids) 
# and intelligently split them (defaulting to 2x1 vertical splits for stacked maps).
# Extracted data and CSV reports are placed in `data/session_<timestamp>/`.
```

### MCP Server (Model Context Protocol)

Palefire implements the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) to expose Ghostwriter capabilities to LLM clients (like Claude Desktop).

```bash
# Run locally (requires environment setup)
python mcp_server.py

# Run via Docker (Recommended)
docker-compose up mcp-server
```

It exposes the following tools:
- `ingest_url`: Download and index content from a URL.
- `ask_question`: RAG-based Q&A.
- `search_content`: Semantic search.
- `list_collections`: List available knowledge collections.

## Episode File Format

Create a JSON file with your episodes:

```json
[
    {
        "content": "Kamala Harris is the Attorney General of California.",
        "type": "text",
        "description": "Biography"
    },
    {
        "content": {
            "name": "Gavin Newsom",
            "position": "Governor",
            "state": "California"
        },
        "type": "json",
        "description": "Structured data"
    }
]
```

See `example_episodes.json` for a complete example.

## Architecture

```
palefire/
├── palefire-cli.py              # Main CLI application
├── modules/                     # Core modules
│   ├── __init__.py
│   ├── PaleFireCore.py         # EntityEnricher + QuestionTypeDetector
│   ├── KeywordBase.py          # Keyword extraction (Gensim)
│   └── api_models.py           # Pydantic models for API
├── agents/                      # AI Agent daemon and parsers
│   ├── AIAgent.py              # ModelManager, AIAgentDaemon
│   ├── palefire-agent-service.py  # Service script
│   ├── parsers/                 # File parsers
│   │   ├── base_parser.py      # Base parser class
│   │   ├── txt_parser.py       # Text file parser
│   │   ├── csv_parser.py       # CSV parser
│   │   ├── pdf_parser.py       # PDF parser
│   │   └── spreadsheet_parser.py  # Excel/ODS parser
│   └── docker-compose.agent.yml  # Docker compose for agent
├── prompts/                     # AI/LLM prompts directory
│   ├── system/                 # System prompts
│   ├── queries/                # Query-related prompts
│   ├── extraction/             # Extraction prompts
│   └── templates/              # Reusable prompt templates
├── examples/                    # Example files for tests
│   ├── input/                  # Test input files
│   └── output/                 # Test output files
├── example_episodes.json        # Example data
├── docs/                        # Documentation folder
│   ├── CLI_GUIDE.md            # Complete CLI documentation
│   ├── QUICK_REFERENCE.md      # Quick reference card
│   ├── ARCHITECTURE.md         # Architecture details
│   └── [other documentation]
└── [other files]
```

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for complete architecture documentation.
See **[prompts/README.md](prompts/README.md)** for prompts organization guide.

## 5-Factor Ranking System

Pale Fire combines 5 independent factors for optimal search results:

1. **Semantic Relevance** (30%) - RRF hybrid search (vector + keyword)
2. **Connectivity** (15%) - How well-connected in the knowledge graph
3. **Temporal Match** (20%) - Active during query time period
4. **Query Term Match** (20%) - Explicit matches of query terms
5. **Entity Type Match** (15%) - Entity types relevant to question type

## Question Types

| Type | Pattern | Boosts | Example |
|------|---------|--------|---------|
| WHO | who, whom, whose | PER (2.0x) | "Who was the AG?" |
| WHERE | where, which place | LOC (2.0x) | "Where did she work?" |
| WHEN | when, what year | DATE (2.0x) | "When was he governor?" |
| WHAT (org) | what organization | ORG (2.0x) | "What organization?" |
| WHAT (position) | what position | PER/ORG (1.5x) | "What position?" |
| HOW MANY | how many | CARDINAL (2.0x) | "How many years?" |
| WHY | why | EVENT (1.5x) | "Why did she leave?" |
| WHAT (event) | what happened | EVENT (2.0x) | "What happened?" |

## Entity Types

Automatically extracted with NER:

- **PER** - Persons (Kamala Harris, Gavin Newsom)
- **LOC** - Locations (California, San Francisco)
- **ORG** - Organizations (Attorney General, FBI)
- **DATE** - Dates (January 3, 2011, 2020)
- **TIME** - Times (3:00 PM, morning)
- **MONEY** - Money ($1 million)
- **PERCENT** - Percentages (50%)
- **EVENT** - Events (World War II)
- Plus 10 more types

## Configuration

All configuration is centralized in `config.py` and loaded from `.env`:

```bash
# Copy example configuration
cp env.example .env

# Edit with your settings
nano .env

# View current configuration
python palefire-cli.py config
```

Key settings:

```bash
# Neo4j (required)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM Provider
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:7b
OLLAMA_VERIFICATION_MODEL=gpt-oss:latest  # Optional: separate model for NER verification

# Search Configuration
DEFAULT_SEARCH_METHOD=question-aware
SEARCH_RESULT_LIMIT=20
SEARCH_TOP_K=5

# Ranking Weights (must sum to ≤ 1.0)
WEIGHT_CONNECTION=0.15
WEIGHT_TEMPORAL=0.20
WEIGHT_QUERY_MATCH=0.20
WEIGHT_ENTITY_TYPE=0.15

# Ghostwriter Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
OLLAMA_HOST=http://localhost:11434
```

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for complete documentation.

## Examples

### Example 1: Political Queries

```bash
# Ingest
python palefire-cli.py ingest --demo

# Query
python palefire-cli.py query "Who was the California Attorney General in 2020?"
python palefire-cli.py query "Where did Kamala Harris work as DA?"
python palefire-cli.py query "When did Gavin Newsom become governor?"
```

### Example 2: Custom Data

```bash
# Create your data file
cat > my_data.json << 'EOF'
[
    {
        "content": "Your content here...",
        "type": "text",
        "description": "Your description"
    }
]
EOF

# Ingest and query
python palefire-cli.py ingest --file my_data.json
python palefire-cli.py query "Your question?"
```

### Example 3: Batch Processing

```bash
# Ingest multiple files
for file in data/*.json; do
    python palefire-cli.py ingest --file "$file"
done

# Run multiple queries
python palefire-cli.py query "Question 1?"
python palefire-cli.py query "Question 2?"
```

## Documentation

All documentation is located in the [`docs/`](docs/) folder. See **[docs/README.md](docs/README.md)** for the complete documentation index.

> **New:** Research documentation now available! See [docs/PROS-CONS.md](docs/PROS-CONS.md) for the theoretical framework and [docs/EVALUATION.md](docs/EVALUATION.md) for evaluation methodology.

### Getting Started
- **[docs/DOCKER.md](docs/DOCKER.md)** - Docker deployment guide (recommended)
- **[docs/PALEFIRE_SETUP.md](docs/PALEFIRE_SETUP.md)** - Manual setup instructions
- **[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Quick reference card
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** - Complete configuration guide

### API & CLI
- **[docs/API_GUIDE.md](docs/API_GUIDE.md)** - REST API documentation
- **[docs/CLI_GUIDE.md](docs/CLI_GUIDE.md)** - Complete CLI documentation

### Features
- **[docs/RANKING_SYSTEM.md](docs/RANKING_SYSTEM.md)** - 5-factor ranking system
- **[docs/NER_ENRICHMENT.md](docs/NER_ENRICHMENT.md)** - NER system guide
- **[docs/QUESTION_TYPE_DETECTION.md](docs/QUESTION_TYPE_DETECTION.md)** - Question-type detection
- **[docs/QUERY_MATCH_SCORING.md](docs/QUERY_MATCH_SCORING.md)** - Query matching details

### Advanced
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Architecture details
- **[docs/REFACTORING_UTILS.md](docs/REFACTORING_UTILS.md)** - Code organization and utils refactoring
- **[docs/TESTING.md](docs/TESTING.md)** - Testing guide and best practices
- **[docs/DATABASE_CLEANUP.md](docs/DATABASE_CLEANUP.md)** - Database cleanup guide
- **[docs/EXPORT_FEATURE.md](docs/EXPORT_FEATURE.md)** - JSON export feature
- **[docs/ENTITY_TYPES_UPDATE.md](docs/ENTITY_TYPES_UPDATE.md)** - Entity types in connections

### Research & Theory
- **[docs/PROS-CONS.md](docs/PROS-CONS.md)** - Pale Fire framework for dataset representation
- **[docs/EVALUATION.md](docs/EVALUATION.md)** - Evaluation framework for interpretive AI systems

### Changelog
- **[docs/CHANGELOG_CONFIG.md](docs/CHANGELOG_CONFIG.md)** - Configuration migration
- **[docs/MIGRATION_SUMMARY.md](docs/MIGRATION_SUMMARY.md)** - Migration summary
- **[docs/EXPORT_CHANGES.md](docs/EXPORT_CHANGES.md)** - Export format changes

## Testing

Pale Fire includes a comprehensive test suite with **126+ tests** covering all major components:

- **Core modules** (EntityEnricher, QuestionTypeDetector)
- **AI Agent** (ModelManager, AIAgentDaemon) - 47 tests
- **File parsers** (TXT, CSV, PDF, Spreadsheet) - 20+ tests
- **API endpoints** and models
- **Search functions** and ranking algorithms
- **Configuration** and utilities

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test suite
pytest tests/test_ai_agent.py -v

# Use test runner script
./run_tests.sh coverage
```

See:
- **[TESTING_SUMMARY.md](TESTING_SUMMARY.md)** - Quick test overview
- **[docs/TESTING.md](docs/TESTING.md)** - Complete testing guide
- **[tests/README.md](tests/README.md)** - Test directory reference

## Requirements

### Core Dependencies
- `graphiti-core>=0.3.0` - Knowledge graph framework
- `python-dotenv>=1.0.0` - Environment variable management
- `gensim>=4.3.0` - Keyword extraction (for keywords command)
- `spacy>=3.7.0` - Named Entity Recognition (optional but recommended)
- `fastapi>=0.104.0` - API framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `websockets>=12.0` - WebSocket server for browser integration
- `youtube-transcript-api>=0.6.0` - YouTube transcript extraction
- `pydantic>=2.5.0` - Data validation

### Optional Dependencies
- `nltk` - For better stemming support in keyword extraction
- `psutil>=5.9.0` - System monitoring for AI Agent daemon
- `PyPDF2>=3.0.0` or `pdfplumber>=0.9.0` - PDF parsing
- `openpyxl>=3.1.0` - Excel .xlsx files
- `xlrd>=2.0.0` - Excel .xls files
- `odfpy>=1.4.0` - OpenDocument Spreadsheet (.ods) files

### Docker (Recommended)
- Docker 20.10+
- Docker Compose 2.0+
- (Optional) NVIDIA Docker for GPU support

### Manual Installation
**Core:**
- Python 3.8+
- graphiti-core
- python-dotenv
- Neo4j database
- websockets>=12.0
- youtube-transcript-api>=0.6.0
- gensim>=4.3.0 (for keyword extraction)

**NER (Optional but Recommended):**
- spacy
- en_core_web_sm model

**Keyword Extraction (Optional but Recommended):**
- gensim>=4.3.0
- nltk (for better stemming support)

**Testing:**
- pytest
- pytest-asyncio
- pytest-cov
- pytest-mock

## Performance

### Without AI Agent (models load each time)
| Operation | Time | Notes |
|-----------|------|-------|
| Model loading | 5-10s | One-time per process |
| Keyword extraction | 0.5-1s | Per request |
| Entity extraction (spaCy) | 50-500ms | Per node |
| Entity extraction (pattern) | 10-50ms | Per node |
| Standard search | 100-300ms | RRF only |
| Question-aware search | 500-2000ms | All factors |

### With AI Agent (models stay loaded)
| Operation | Time | Notes |
|-----------|------|-------|
| Model loading | 5-10s | One-time on daemon startup |
| Keyword extraction | 0.01-0.1s | **10-100x faster!** |
| Entity extraction (spaCy) | 50-500ms | Same as above |
| File parsing | Varies | Depends on file type and size |
| Standard search | 100-300ms | RRF only |
| Question-aware search | 500-2000ms | All factors |

### Question Detection
| Operation | Time | Notes |
|-----------|------|-------|
| Question detection | 1-5ms | Regex-based |

## Troubleshooting

### Module not found
```bash
cd /path/to/palefire
python palefire-cli.py --help
```

### spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

### Gensim not found (for keyword extraction)
```bash
pip install gensim>=4.3.0
# Optional: For better stemming support
pip install nltk
```

### File parsing dependencies missing
```bash
# Install all parsing dependencies
pip install PyPDF2>=3.0.0 openpyxl>=3.1.0 xlrd>=2.0.0 odfpy>=1.4.0 requests>=2.31.0 beautifulsoup4>=4.12.0

# Or install individually as needed
pip install PyPDF2>=3.0.0  # For PDF files
pip install openpyxl>=3.1.0  # For .xlsx files
pip install requests>=2.31.0 beautifulsoup4>=4.12.0  # For URL/HTML parsing
pip install xlrd>=2.0.0  # For .xls files
pip install odfpy>=1.4.0  # For .ods files
```

### AI Agent daemon not starting
```bash
# Check if daemon is already running
python palefire-cli.py agent status

# Check logs
tail -f logs/palefire-agent.log

# Verify dependencies
pip install psutil>=5.9.0
```

### Neo4j connection error
```bash
# Check Neo4j is running
# Verify credentials in .env
```

## Best Practices

1. ✅ **Use AI Agent Daemon** for production - eliminates model loading delays
2. ✅ Use NER enrichment for production
3. ✅ Use question-aware search for natural questions
4. ✅ Batch process large datasets
5. ✅ Monitor logs for errors
6. ✅ Backup Neo4j database regularly
7. ✅ **Keep daemon running** - models stay loaded, requests are instant
8. ✅ **Parse files once** - reuse parsed text for multiple operations
9. ✅ Use appropriate parsers - PDF parsers vary in speed (pdfplumber slower but better)

## AI Agent Daemon

The AI Agent daemon keeps Gensim and spaCy models loaded in memory to avoid start/stop delays. This is especially useful for production deployments with high request volumes.

### Features

- **⚡ Fast Access**: Models stay loaded, eliminating 5-10 second initialization delays
- **🔄 Thread-Safe**: Safe concurrent access to models via ModelManager
- **📄 File Parsing**: Integrated parsers for TXT, CSV, PDF, and spreadsheet files
- **🔑 Keyword Extraction**: Fast keyword and n-gram extraction with configurable methods
- **🏷️ Entity Extraction**: Instant NER extraction using loaded spaCy models
- **📊 Status Monitoring**: Real-time status with process information (PID, memory, CPU)

### Quick Start

```bash
# Start daemon in background
python palefire-cli.py agent start --daemon

# Check status (shows PID, memory, CPU usage)
python palefire-cli.py agent status

# Stop daemon
python palefire-cli.py agent stop

# Restart daemon
python palefire-cli.py agent restart --daemon
```

### Using the Daemon Programmatically

```python
from agents import get_daemon

# Get daemon instance (models loaded once)
daemon = get_daemon(use_spacy=True)
daemon.model_manager.initialize(use_spacy=True)

# Extract keywords (fast - models already loaded)
keywords = daemon.extract_keywords(
    "Your text here",
    num_keywords=10,
    method='combined',
    enable_ngrams=True,
    min_ngram=2,
    max_ngram=3
)

# Extract entities (fast - models already loaded)
entities = daemon.extract_entities("Your text here")

# Parse files
result = daemon.parse_file("document.pdf")
if result['success']:
    text = result['text']
    metadata = result['metadata']
```

### Automatic Daemon Management

The `keywords` command automatically checks if the daemon is running and starts it if needed:

```bash
# This will start the daemon automatically if not running
python palefire-cli.py keywords "Your text here"
```

### Docker Deployment

**Standalone:**
```bash
# Start the AI Agent daemon
docker-compose -f agents/docker-compose.agent.yml up -d

# View logs
docker-compose -f agents/docker-compose.agent.yml logs -f

# Stop the agent
docker-compose -f agents/docker-compose.agent.yml down
```

**Integrated with main services:**
```bash
# Start all services including the agent
docker-compose -f docker-compose.yml -f agents/docker-compose.agent.yml up -d
```

See **[agents/DOCKER.md](agents/DOCKER.md)** for complete Docker documentation.

See **[agents/USAGE_GUIDE.md](agents/USAGE_GUIDE.md)** for complete usage guide on starting, stopping, and querying the agent.

### System Service Integration

**Linux (systemd):**
```bash
# Copy service file
sudo cp agents/palefire-agent.service /etc/systemd/system/
# Edit paths in service file
sudo nano /etc/systemd/system/palefire-agent.service
# Enable and start
sudo systemctl enable palefire-agent
sudo systemctl start palefire-agent
```

**macOS (launchd):**
```bash
# Copy plist file
cp agents/palefire-agent.plist ~/Library/LaunchAgents/
# Edit paths in plist file
nano ~/Library/LaunchAgents/palefire-agent.plist
# Load service
launchctl load ~/Library/LaunchAgents/palefire-agent.plist
```

### File Parsing Capabilities

The AI Agent includes integrated file parsers for extracting text from various formats:

- **TXT**: Plain text files with encoding detection
- **CSV**: Comma-separated values with delimiter auto-detection
- **PDF**: Text and table extraction (PyPDF2 or pdfplumber)
- **Spreadsheets**: Excel (.xlsx, .xls) and OpenDocument (.ods) with multi-sheet support
- **URL/HTML**: Extract text from web pages using BeautifulSoup with script/style removal

```python
from agents import get_daemon

daemon = get_daemon()
result = daemon.parse_file("document.pdf", max_pages=10)

# Result structure:
# {
#     'text': 'Full extracted text...',
#     'metadata': {'filename': 'document.pdf', 'page_count': 5, ...},
#     'pages': ['Page 1 text...', 'Page 2 text...'],
#     'tables': [{'data': [...], 'headers': [...]}],
#     'success': True,
#     'error': None
# }
```

### Benefits

- **⚡ No Model Loading Delays**: Models stay in memory, ready for instant use (10-100x faster!)
- **🔄 Reduced Memory Overhead**: Single instance shared across requests
- **📈 Better Performance**: Eliminates repeated model initialization
- **🏭 Production Ready**: Designed for high-throughput scenarios
- **📄 Unified Interface**: Single daemon handles keywords, entities, and file parsing

## Future Enhancements

- [x] REST API wrapper (see [docs/API_GUIDE.md](docs/API_GUIDE.md))
- [x] AI Agent daemon for model persistence
- [x] File parsers (TXT, CSV, PDF, Spreadsheet)
- [x] Keyword extraction with n-grams
- [x] Comprehensive unit tests for AI Agent (47+ tests)
- [ ] Web UI
- [ ] Result caching
- [ ] Multi-language support
- [ ] Custom entity types
- [ ] ML-based question detection
- [ ] Socket/HTTP communication for daemon
- [ ] Additional file formats (DOCX, RTF, etc.)

## Contributing

When adding features:
1. Add classes to `modules/PaleFireCore.py`
2. Add functions to `palefire-cli.py`
3. Update documentation
4. Test thoroughly

## License

Inherits license from parent Open WebUI project.

## Support

For issues or questions:
1. Check documentation files in [docs/](docs/)
2. Review [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md)
3. Check logs for error messages
4. Verify environment configuration

---

**Pale Fire** - Intelligent Knowledge Graph Search Made Easy 🚀
