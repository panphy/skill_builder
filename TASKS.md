# Improvement Tasks

## Task 1: Graceful startup without secrets (completed)
- [x] Use safe secret accessors to avoid crashes when `secrets.toml` is missing.
- [x] Ensure database/Supabase/OpenAI lookups fall back cleanly to empty values.

## Task 2: Server-side filtering and pagination (completed)
- [x] Add database queries for distinct topic/sub-topic/skill/difficulty lists.
- [x] Load questions with filters applied in SQL instead of client-side filtering.
- [x] Implement pagination controls (page size + page index) in the student view.
