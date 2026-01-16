# Improvement Tasks

## Task 1: Graceful startup without secrets (completed)
- [x] Use safe secret accessors to avoid crashes when `secrets.toml` is missing.
- [x] Ensure database/Supabase/OpenAI lookups fall back cleanly to empty values.

## Task 2: Server-side filtering and pagination (planned)
- [ ] Add database queries for distinct topic/sub-topic/skill/difficulty lists.
- [ ] Load questions with filters applied in SQL instead of client-side filtering.
- [ ] Implement pagination controls (page size + page index) in the student view.
