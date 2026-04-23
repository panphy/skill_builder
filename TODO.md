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

### A1 — Add timeout to LLM question-generation API calls ✅ DONE
- [x] `ai_generation.py` — Added `client.with_options(timeout=60)` to the `_call_model` inner function in `generate_question` and `client.with_options(timeout=90)` in `generate_journey` (journey uses `max_completion_tokens=4000` so needs more time). Pattern matches the existing feedback call at line 299.
- **Files touched**: `ai_generation.py`

### A2 — Fix PIL image memory leak in compression loop ✅ DONE
- [x] `image_utils.py` — In the resize-and-compress loop, `img2` is now set to `None` when adopted (`img = img2; img2 = None`) and explicitly closed via `img2.close()` when discarded. This releases memory promptly instead of waiting for GC.
- **Files touched**: `image_utils.py`

### A3 — Audit `unsafe_allow_html=True` usage ✅ DONE (no code change needed)
- [x] `app.py:215` — Inline `<script>` for track localStorage restore. Content is JSON-encoded system constants only. **Safe.**
- [x] `app.py:231` — Inline `<script>` to persist track. `track_value` is validated against `TRACK_ALLOWED` whitelist before use. **Safe.**
- [x] `app.py:1532` — Logo `<img>` tag using hardcoded `PANPHY_LOGO_URL` constant. **Safe.**
- [x] `app.py:1603` — Footer with hardcoded HTML strings. **Safe.**
- **Result**: All 4 usages render only system-controlled or constant content. No user input reaches any of these call sites. No changes required.

### A4 — Add server-side image type validation (magic bytes) ✅ DONE
- [x] `image_utils.py` — Added `_PNG_MAGIC`, `_JPEG_MAGIC` constants and `_has_valid_image_magic()` helper. Both `validate_image_file` and `_compress_bytes_to_limit` now reject files with invalid magic bytes before calling `Image.open()`. Gives a clear error message and avoids unnecessary PIL processing.
- **Files touched**: `image_utils.py`

### A5 — Sanitize database error messages shown to users ✅ DONE
- [x] All `db_last_error` assignments that previously included `{e}` (the raw exception string) have been changed to include only `{type(e).__name__}`. Full tracebacks are now logged server-side via `LOGGER.exception(...)`.
- [x] Locations fixed: `app.py` (7 sites — table creation, insert, storage upload/download, load, delete) and `db.py` (2 sites — engine creation, question bank load).
- [x] Locations where `LOGGER.error(...)` existed were upgraded to `LOGGER.exception(...)` to capture full tracebacks. Locations with no prior logger call had one added.
- **Files touched**: `app.py`, `db.py`

### A6 — Add correlation IDs to structured logging ✅ DONE
- [x] `app.py` — Added `_ss_init("session_id", pysecrets.token_hex(6))` to session state init (12-char hex, unique per browser session).
- [x] `app.py` — Added `SessionIDFilter` class that automatically injects `[sid=...]` into every log record's `ctx` dict at log time. Filter is registered on the `panphy` logger in `setup_logging()`. No changes required to individual log call sites.
- **Files touched**: `app.py`

### A7 — Lower default row limit for `load_question_bank_df` ✅ DONE
- [x] `db.py:306` — Default `limit` changed from `5000` to `1000`. The only existing caller (`ui_teacher.py:281`) already passes `limit=5000` explicitly so teacher behaviour is unchanged. The new default protects against future callers that omit the argument.
- **Files touched**: `db.py`

---

## Group B — Refactor (Single File)

### B1 — Extract duplicated canvas rendering into a shared helper ✅ DONE
- [x] `ui_student.py` — The canvas rendering block (tool controls + canvas widget) was duplicated between single-question mode (~lines 606–696) and journey mode (~lines 968–1065).
- [x] Extracted into `_render_canvas(slot, canvas_height, canvas_storage_key, qid, step_i=None)` nested function inside `render_student_page`, added after `_render_filter_chips`.
- [x] Both blocks replaced with `canvas_value, canvas_result = _render_canvas(...)` one-liners.
- [x] `slot` parameter ("single" or "journey") drives all key names; `step_i` is included in the stylus canvas key only when present (journey mode).
- **Note**: `ui_teacher.py` has no canvas code. No `ui_shared.py` needed — this was a single-file refactor.
- **Files touched**: `ui_student.py` only.

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

| Task | Independent? | Files Touched | Status |
|------|-------------|---------------|--------|
| A1 — LLM timeout | Yes | `ai_generation.py` | ✅ Done |
| A2 — PIL memory leak | Yes | `image_utils.py` | ✅ Done |
| A3 — XSS / unsafe HTML | Yes | `app.py` (audit only) | ✅ Done |
| A4 — Magic bytes validation | Yes | `image_utils.py` | ✅ Done |
| A5 — Sanitize error messages | Yes | `app.py`, `db.py` | ✅ Done |
| A6 — Correlation IDs | Yes | `app.py` | ✅ Done |
| A7 — Query row limit | Yes | `db.py` | ✅ Done |
| B1 — Extract canvas helper | Single file | `ui_student.py` | ✅ Done |
| C1 — Refactor `app.py` | After all A + B tasks | `app.py` + new modules | ⬜ Todo |
| C2 — Token bucket rate limit | After C1 | `rate_limiter.py`, `db.py` | ⬜ Todo |

**Recommended order if tackling one at a time**: ~~A1~~ ~~A2~~ ~~A3~~ ~~A4~~ ~~A5~~ ~~A6~~ ~~A7~~ ~~B1~~ → C1 → C2
