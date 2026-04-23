# AGENTS.md - AI Assistant Guide for PanPhy Skill Builder

This document provides essential context for AI assistants working on this codebase.

## Project Overview

PanPhy Skill Builder is a **Streamlit web application for GCSE Physics practice** that combines:
- AI-generated exam-style questions with mark schemes
- Teacher-uploaded scanned questions and mark schemes (stored in Supabase Storage)
- Student workspace with typed or handwritten (canvas) answer modes
- AI feedback with Markdown and LaTeX rendering
- Topic Journey mode for scaffolded step-by-step learning
- Combined vs. Separate track filtering for GCSE Physics vs. Trilogy Physics
- Per-student rate limiting for AI submissions

This is a **production classroom application** - changes should prioritize stability and real-world usability.

## Quick Reference

```bash
# Run the app
streamlit run app.py

# Run tests
python -m unittest discover tests

# Default port: 8501
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend/App | Streamlit |
| AI/LLM | OpenAI API (GPT-5-mini) |
| Database | Supabase Postgres + SQLAlchemy 2.0+ |
| File Storage | Supabase Storage |
| Image Processing | Pillow |
| Canvas | Custom `panphy_stylus_canvas` component |

## Repository Structure

```
skill_builder/
├── app.py                    # Main entry point, routing, constants, logging setup
├── ui_student.py             # Student interface and workflow
├── ui_teacher.py             # Teacher dashboard and content management
├── ai_generation.py          # LLM prompts, validation, journey generation
├── db.py                     # Database layer (SQLAlchemy, all SQL queries)
├── config.py                 # Subject pack loading, topic catalogs
├── image_utils.py            # Image compression, validation, format conversion
├── components/
│   └── panphy_stylus_canvas/ # Custom Streamlit drawing canvas component
├── subjects/
│   └── physics/
│       ├── topics.json       # GCSE Physics topics catalog
│       ├── prompts.json      # AI prompt templates
│       ├── equations.json    # Equation whitelist and notation rules
│       └── settings.json     # Subject display config
├── utils/
│   └── json_utils.py         # Safe JSON parsing from LLM responses
├── tests/
│   ├── test_json_utils.py    # JSON parsing tests
│   └── test_image_processing.py  # Image handling tests
├── requirements.txt          # Python dependencies
├── specs.md                  # Project conventions
└── README.md                 # User documentation
```

## Key Files and Their Purposes

| File | Purpose | When to Modify |
|------|---------|----------------|
| `app.py` | Entry point, constants, logging, session state | Adding global constants, new pages, logging changes |
| `ui_student.py` | Student question selection, answer submission, feedback display | Student-facing features |
| `ui_teacher.py` | Question bank management, uploads, journey authoring | Teacher-facing features |
| `ai_generation.py` | All LLM interactions, prompt templates, validation | AI behavior, prompt engineering |
| `db.py` | All database operations, schema, caching | Database schema changes, new queries |
| `config.py` | Configuration loaders, topic/prompt accessors | Configuration loading logic |
| `subjects/physics/*.json` | Subject-specific content and prompts | Content changes, new prompts |

## Development Conventions

### Code Style
- **Small, focused changes** - Avoid large refactors unless explicitly requested
- **Logic placement** - UI code in `ui_*.py`, shared utilities in `utils/`, database in `db.py`
- **Error handling** - Handle expected errors explicitly, avoid broad exception catching
- **Comments** - Only add when they provide clarity beyond the code itself
- **No over-engineering** - Don't add features, abstractions, or improvements beyond what was asked

### Naming Conventions
- Database tables: `_v2` suffix for versioned schema (e.g., `question_bank_v2`)
- Session state keys: `st.session_state["snake_case_key"]`
- Private functions: `_underscore_prefix()`
- Cache key parameter: `_fp` = first 40 chars of DB URL (for cache invalidation)

### Streamlit Patterns
- Use `@st.cache_data` for data queries with TTL
- Use `@st.cache_resource` for expensive resources (DB engines)
- Initialize session state with `_ss_init(key, default_value)` helper in `app.py`
- UI locking during AI operations uses overlay with progress bar

### Prompt Templates
- Token replacement syntax: `<<TOKEN_NAME>>`
- Stored in `subjects/physics/prompts.json`
- Rendered via `_render_template()` in `ai_generation.py`

## Database Schema

Three main tables (auto-created on startup):

### question_bank_v2
```sql
- id (bigserial PK)
- source (teacher | ai_generated)
- topic, sub_topic, skill, difficulty
- question_text, question_image_path
- markscheme_text, markscheme_image_path
- question_type (single | journey)
- journey_json (jsonb) - for Topic Journey steps
- track_ok (both | separate_only)
- is_active (boolean)
```

### attempts_v2
```sql
- id (bigserial PK)
- student_id (hashed for anonymity)
- question_bank_id, step_index
- mode (typed_response | handwritten | journey)
- marks_awarded, max_marks
- feedback_points, next_steps (jsonb)
```

### rate_limits_v2
```sql
- student_id (text PK)
- submission_count (int)
- window_reset_at (timestamptz)
```

## Configuration System

### Environment Variables / Secrets
```
SUBJECT_SITE     # Which subject pack to load (default: "physics")
DATABASE_URL     # Supabase PostgreSQL connection string
OPENAI_API_KEY   # OpenAI API credentials
PANPHY_LOG_FILE  # Override log file path (default: panphy_app.log)
```

Access secrets safely using `_safe_secret(key, default)` from `config.py`.

### Subject Pack Files (subjects/physics/)
| File | Purpose |
|------|---------|
| `topics.json` | GCSE curriculum topics with AQA spec refs |
| `prompts.json` | AI prompt templates for generation, journey, feedback |
| `equations.json` | Equation whitelist (only AQA-approved equations) |
| `settings.json` | Display config (question types, difficulties, skills) |

## Key Implementation Patterns

### AI Question Generation
1. Template-based prompts with equation whitelist enforcement
2. LLM self-checking for errors and clarity
3. Repair prompts if validation fails (without regenerating)
4. Forbidden content checks (A-level, chemistry, off-spec)

### AI Feedback Marking
1. Receives question + student answer (text or image) + mark scheme
2. Confidentiality rule: mark scheme never revealed in feedback
3. Outputs structured JSON: marks, summary, feedback points, next steps

### Rate Limiting
- Per-student (or anonymous ID) - tighter limits for anon users
- Window-based (default: 10 submissions per hour)
- Checked in `db.py` before each AI feedback request

### Image Handling
- Compression with quality/resolution scaling via `image_utils.py`
- Max dimensions: 4000x4000 px
- Size limits: Questions 5MB, Mark schemes 5MB, Canvas 2MB
- Transparent PNG to JPEG conversion with white background

## Testing

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_json_utils
```

Test files:
- `tests/test_json_utils.py` - JSON parsing robustness
- `tests/test_image_processing.py` - Image compression and format conversion

**Convention**: Update or add tests when behavior changes. If tests are not run, note the reason.

## Common Tasks

### Adding a New Topic
1. Edit `subjects/physics/topics.json`
2. Add topic object with `name`, `group`, `track_ok`, `aqa_spec_ref`

### Modifying AI Prompts
1. Edit `subjects/physics/prompts.json`
2. Use `<<TOKEN_NAME>>` for replaceable values
3. Test with the teacher preview feature before saving questions

### Adding a Database Column
1. Modify table definition in `db.py` (look for `CREATE TABLE` statements)
2. Add column with `IF NOT EXISTS` or use `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
3. Update related queries and any cached dataframe calls

### Modifying Student/Teacher UI
1. Edit `ui_student.py` or `ui_teacher.py`
2. Follow existing patterns for expanders, columns, and session state
3. Use `st.session_state` for component state persistence

## Important Constants (app.py)

```python
RATE_LIMIT_MAX = 10                    # Submissions per window
RATE_LIMIT_WINDOW_SECONDS = 3600       # 1 hour window
QUESTION_MAX_MB = 5.0                  # Question image size limit
MARKSCHEME_MAX_MB = 5.0                # Mark scheme size limit
CANVAS_MAX_MB = 2.0                    # Canvas submission size limit
STORAGE_BUCKET = "physics-bank"        # Supabase storage bucket
```

## Logging

Uses structured logging with `KVFormatter`:
```python
LOGGER.info("Message", extra={"ctx": {"component": "db", "action": "query"}})
```

Log file: `panphy_app.log` (rotating, 2MB max, 3 backups)

## Checklist Before Making Changes

1. [ ] Read the relevant source files before modifying
2. [ ] Check `specs.md` for any specific conventions
3. [ ] Consider impact on both student and teacher views
4. [ ] Ensure changes work with both Combined and Separate tracks
5. [ ] Test with missing secrets/database (graceful degradation)
6. [ ] Run tests if modifying core logic: `python -m unittest discover tests`
7. [ ] Avoid introducing security vulnerabilities (XSS, SQL injection)

## What NOT to Do

- Don't add features beyond what was requested
- Don't refactor surrounding code when fixing a bug
- Don't add type annotations or docstrings to unchanged code
- Don't create abstractions for one-time operations
- Don't add error handling for impossible scenarios
- Don't reveal mark schemes in AI feedback (confidentiality rule)
