# Improvement Tasks

## Task 1: Graceful startup without secrets (completed)
- [x] Use safe secret accessors to avoid crashes when `secrets.toml` is missing.
- [x] Ensure database/Supabase/OpenAI lookups fall back cleanly to empty values.

## Task 2: Server-side filtering and pagination (completed)
- [x] Add database queries for distinct topic/sub-topic/skill/difficulty lists.
- [x] Load questions with filters applied in SQL instead of client-side filtering.
- [x] Implement pagination controls (page size + page index) in the student view.

## Task 3: UI/UX polish for Streamlit views (completed)
- [x] Add a persistent top-level navigation hint (e.g., a short “Start here” callout in the hero) that clarifies the student vs. teacher workflow before the first expander opens.
- [x] Group student filters into a compact “Filters” card with a quick “Reset filters” button and visible active filter chips.
- [x] Add a lightweight “Question status” block (selected source + topic + difficulty + question count) so students can confirm their context before answering.
- [x] Provide clearer empty/error states for missing topics or DB configuration with actionable next steps and links to the teacher upload workflow.
- [x] Improve scanability of teacher dashboards by replacing raw dataframes with tabbed summaries (Overview / By student / By topic / Recent), keeping the detailed table expandable.
- [x] Add inline helper text to teacher upload/generation forms that clarifies required fields, accepted image sizes, and expected naming conventions.
- [x] Normalize spacing and section headings between student and teacher pages so that switching roles feels consistent (matching header hierarchy and divider placement).
