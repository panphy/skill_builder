# PanPhy Skill Builder

A Streamlit web app for GCSE Physics practice that combines:
- AI-generated exam-style questions with mark schemes
- Teacher-uploaded scanned questions and mark schemes
- A student workspace for typed answers or handwritten working on a canvas
- AI feedback that is rendered properly in Markdown and LaTeX
- A Topic Journey mode for guided step-by-step practice
- A track selector (Combined vs Separate) to filter topics by eligibility
- Per-student rate limiting for AI submissions

This repo is built to be practical for real classroom use: fast to deploy, simple to maintain, and designed to reduce “messing around” during submission by locking the UI while the AI is processing.

---

## What this app does

### Student view
Students can:
- Select questions from either:
  - **AI Practice** (AI-generated questions saved to the database)
  - **Teacher Upload** (scanned images stored in Supabase Storage)
- Filter question bank results with server-side topic/skill/difficulty filters and paginated pages
- Answer using:
  - **Typed response**
  - **Handwritten response** via a drawing canvas
- Click **Submit** and receive AI feedback (Markdown + LaTeX supported)
- Use **Topic Journey** for scaffolded, step-by-step learning
- Switch between **Combined** (core-only) and **Separate** (full) tracks to filter question visibility

While the AI is working, the UI is locked with a translucent overlay and a progress bar to prevent accidental changes.

To keep submissions fair and responsive, AI feedback requests are rate-limited per student ID, with tighter limits applied to anonymous users (default: 10 submissions per hour).

### Question quality safeguards
To improve consistency and correctness in AI-generated questions, the app enforces:
- **Equation whitelist**: Only approved equations are used in question generation, keeping problems aligned with the GCSE Physics specification.
- **LLM self-checking**: The model performs an internal validation pass to catch errors, verify working, and improve clarity before a question is saved.

### Teacher view
Teachers can:
- Browse and preview the question bank (AI + teacher uploads)
- Generate AI question drafts, edit them, and save them to the database (teacher vetting is compulsory for quality assurance)
- Upload scanned question and mark scheme images to storage and index them in the database
- Create Topic Journey assignments (fixed at ~10 minutes / 5 steps) and save them as a single bank entry
- Control visibility for Combined vs Separate-only topics when saving new content

---

## Tech stack

- **Frontend / App**: Streamlit
- **AI**: OpenAI API (default model set in code)
- **Database**: Supabase Postgres
- **File storage**: Supabase Storage (for scanned images)
- **Python libraries**: SQLAlchemy, psycopg, Pillow, pandas, streamlit-drawable-canvas
- **Logging**: Stream + rotating log file (`panphy_app.log`, created at runtime in the working directory; override with `PANPHY_LOG_FILE`)

---

## Repo structure

- `app.py`  
  Main Streamlit app.
- `components/`
  Custom Streamlit component(s) for the stylus/drawing canvas.
  - `panphy_stylus_canvas/`
    Component package with the HTML/JS frontend and Python wrapper.
- `ui_student.py`
  Student page layout and interaction logic.
- `ui_teacher.py`
  Teacher dashboard for question bank, uploads, and Topic Journey authoring.
- `ai_generation.py`
  LLM prompt orchestration, equation whitelist checks, and Topic Journey generation.
- `config.py`
  Subject pack loading, topic catalogs, and prompt settings.
- `db.py`
  Database access and migrations for attempts, question bank, and rate limits.
- `subjects/`
  Subject configuration and content data.
  - `physics/`
    Physics prompts, equations, settings, and topic definitions (JSON).
- `requirements.txt`  
  Python dependencies.
- `README.md`  
  This file.

---
