# Project Specs

This document captures conventions and project context to guide agentic tooling and contributors.

## Purpose
- Provide a shared source of truth for workflows, quality checks, and code style.
- Reduce ambiguity when making changes or adding features.

## Repository Map (high level)
- `app.py`: application entry point and wiring.
- `ui_student.py`, `ui_teacher.py`: UI flows for student/teacher experiences.
- `components/`: UI components.
- `subjects/`: subject-specific content and logic.
- `utils/`, `image_utils.py`, `ai_generation.py`: shared helpers and generation utilities.
- `db.py`: data access layer.
- `tests/`: test suite.

## Development Workflow
- Prefer small, focused changes with clear diffs.
- Keep logic close to the feature area (UI in `ui_*.py`, shared utilities in `utils/`).
- Update or add tests when behavior changes.

## Code Style
- Use clear, descriptive names and keep functions focused.
- Avoid broad exception handling; handle expected errors explicitly.
- Document behavior in code comments only when it adds clarity beyond the code itself.

## Testing
- Run relevant tests when modifying behavior.
- If tests are not run, note the reason in the summary.

## Agent Guidelines
- Read this file before making changes.
- When unsure about placement, search for similar patterns in nearby files.
