# TODO — AI Agent Task List

Each task is self-contained unless a **dependency note** says otherwise.
Tasks marked **INDEPENDENT** can be assigned to any agent individually without risk of conflict.
Tasks in a **GROUP** must be done as a unit (they touch overlapping code).
Tasks in a **SEQUENCE** must follow a specific order but can be done separately over time.

---

## Group A — Quick Independent Fixes
> All tasks below are **INDEPENDENT** of each other.
> An agent can be given any one of these tasks without touching any other task.
> Safe to tackle one at a time, in any order.

### A1 — Add timeout to LLM question-generation API calls
- [ ] `ai_generation.py:542` — `client.chat.completions.create(...)` has no timeout. The feedback call at line 299 already uses `client.with_options(timeout=10)` as the correct pattern. Apply the same `with_options(timeout=60)` (generation is slower than feedback) to the `_call_model` inner functions inside `generate_question` (~line 502) and `generate_journey` (~line 729).
- **Risk if skipped**: A slow/hung OpenAI API response blocks the entire Streamlit thread indefinitely with no recovery.
- **Files touched**: `ai_generation.py` only.

### A2 — Fix PIL image memory leak in compression loop
- [ ] `image_utils.py:123` — Inside the resize-and-compress loop, `img2 = img.resize(...)` creates new PIL Image objects on each iteration that are never explicitly closed. Wrap them in a `try/finally` block or use the image as a context manager to call `.close()` when no longer needed.
- **Risk if skipped**: Memory grows on high-volume upload sessions (low urgency in classroom setting).
- **Files touched**: `image_utils.py` only.

### A3 — Audit and restrict `unsafe_allow_html=True` usage
- [ ] `app.py:215` — Audit: confirm this call only renders AI-generated or system-controlled content (never raw student/teacher input).
- [ ] `app.py:231` — Same audit.
- [ ] `app.py:1532` — Same audit.
- [ ] `app.py:1603` — Same audit.
- [ ] For any call site where user-controlled text (topic names, question labels, student answers) could reach the rendered string, switch to `unsafe_allow_html=False` or pre-escape HTML entities before rendering.
- **Risk if skipped**: If any user-controlled text reaches these call sites, stored XSS is possible.
- **Files touched**: `app.py` only.

### A4 — Add server-side image type validation (magic bytes)
- [ ] `ui_teacher.py:1296–1297` — `st.file_uploader(..., type=["png","jpg","jpeg"])` only checks the filename extension. After reading the uploaded bytes, call `image_utils.validate_and_compress_image()` (which calls `Image.open()`) — this already catches invalid files at the PIL layer. To close the gap fully, add an explicit check of the first 3–4 bytes against known PNG (`\x89PNG`) and JPEG (`\xff\xd8\xff`) magic bytes before passing to PIL.
- **Note**: Low urgency — `Image.open()` in `image_utils.py:91` already rejects non-image bytes and returns an error. This task tightens the validation layer.
- **Files touched**: `ui_teacher.py` or `image_utils.py` only.

### A5 — Sanitize database error messages shown to users
- [ ] Throughout `app.py` and `db.py`, caught exceptions are sometimes surfaced directly to the UI (e.g. `st.error(str(e))`). Replace with a generic user message ("Something went wrong — please try again") while keeping the full exception in the server-side log via `LOGGER.exception(...)`.
- [ ] Specifically check: `app.py:410–412`, `db.py:61–62`, and any `st.error(...)` call that includes `e` or `str(e)` directly.
- **Risk if skipped**: Database schema, table names, or connection strings can appear in the UI on errors.
- **Files touched**: `app.py`, `db.py` (string-level changes only, no logic changes).

### A6 — Add correlation IDs to structured logging
- [ ] In `app.py`, at the start of each user-triggered action (submit answer, generate question, upload file), generate a short UUID (`uuid.uuid4().hex[:8]`) and store it in `st.session_state["request_id"]`.
- [ ] Pass it into log calls via `extra={"ctx": {"request_id": ..., ...}}` so all log lines for one user action share the same ID.
- **Risk if skipped**: Hard to trace a single user's flow through multi-step log output.
- **Files touched**: `app.py` primarily; minor additions to `ai_generation.py` and `db.py` log calls.

### A7 — Lower default row limit for `load_question_bank_df`
- [ ] `db.py:306` — Default `limit=5000` is passed from `app.py`. Add a query execution safeguard: if the question bank grows large, this full load will slow page renders. Reduce the default to `1000` and confirm no caller depends on loading all rows silently. Add a comment explaining the limit.
- **Files touched**: `db.py`, possibly `app.py` (where `load_question_bank_df` is called with no explicit limit).

---

## Group B — Coordinated Refactor (Two Files, One Task)
> **DEPENDENCY**: Tasks B1 must be done as a single unit.
> Do NOT split B1 across two agents or interleave it with any Group C work.
> It is safe to do B1 before or after any Group A task.

### B1 — Extract shared canvas and filter UI into a shared helper
- [ ] `ui_student.py` and `ui_teacher.py` both contain near-identical canvas rendering blocks and question filter UI. Extract these into a new `ui_shared.py` module (or add to an existing shared utility file).
- [ ] Canvas rendering: identify the duplicated block in `ui_student.py` (~lines 639–696) and `ui_teacher.py` (the corresponding block), and replace both with a call to a shared `render_canvas(key_prefix)` helper.
- [ ] Filter UI: identify duplicated filter widget logic across both files and extract into a shared `render_question_filters(...)` helper.
- [ ] After extraction, run `python -m unittest discover tests` to confirm nothing broke.
- **Why one task**: Both files must be updated together. If only one is updated, the extracted helper won't exist for the other, causing an import error.
- **Files touched**: `ui_student.py`, `ui_teacher.py`, new `ui_shared.py`.

---

## Group C — Large Structural Refactor (Do Last, Do as a Unit)
> **SEQUENCE DEPENDENCY**: Complete all Group A and Group B tasks first.
> Group C touches `app.py` which overlaps with A3, A5, and A6.
> Doing C before A means merge conflicts or duplicated effort.
> Do NOT interleave C with any other task.

### C1 — Break up `app.py` into focused modules
`app.py` is 1604 lines and mixes six unrelated concerns. Extract each into its own module:

- [ ] **`storage.py`** — Move all Supabase Storage upload/download logic out of `app.py`. Functions: anything calling `supabase.storage`.
- [ ] **`rate_limiter.py`** — Move rate-limit check and increment logic (~`app.py:550–639`) into a standalone module. This also makes it independently testable.
- [ ] **`session_state.py`** — Move all `_ss_init(...)` calls and session-state setup into an explicit initialization module called once at startup.
- [ ] Keep `app.py` as a thin orchestration layer: page config, routing, and `main()`. Target < 400 lines after extraction.
- [ ] After each extraction step, run `streamlit run app.py` and verify the app loads correctly before moving to the next.
- [ ] Run `python -m unittest discover tests` after all extractions are done.

### C2 — Improve rate limiting to prevent burst abuse
> **SEQUENCE DEPENDENCY**: Do after C1 (rate limit logic will have been moved to `rate_limiter.py`).

- [ ] Replace the current fixed-window reset (resets at submission time on boundary) with a sliding-window or token-bucket implementation.
- [ ] Ensure the rejection response is explicit to the user — currently the limit may silently truncate without a clear UI message.
- [ ] Consider adding a per-IP limit in addition to per-student-ID, especially for anonymous users.
- **Files touched**: `rate_limiter.py` (after C1), `db.py` (schema changes to `rate_limits_v2` if needed).

---

## Summary Table

| Task | Independent? | Files Touched | Effort |
|------|-------------|---------------|--------|
| A1 — LLM timeout | Yes | `ai_generation.py` | Small |
| A2 — PIL memory leak | Yes | `image_utils.py` | Small |
| A3 — XSS / unsafe HTML | Yes | `app.py` | Small |
| A4 — Magic bytes validation | Yes | `ui_teacher.py` | Small |
| A5 — Sanitize error messages | Yes | `app.py`, `db.py` | Small |
| A6 — Correlation IDs | Yes | `app.py`, others | Small |
| A7 — Query row limit | Yes | `db.py`, `app.py` | Tiny |
| B1 — Extract shared UI | Must be done as one unit | `ui_student.py`, `ui_teacher.py`, new file | Medium |
| C1 — Refactor `app.py` | After all A + B tasks | `app.py` + new modules | Large |
| C2 — Token bucket rate limit | After C1 | `rate_limiter.py`, `db.py` | Medium |

**Recommended order if tackling one at a time**: A1 → A2 → A3 → A4 → A5 → A6 → A7 → B1 → C1 → C2
