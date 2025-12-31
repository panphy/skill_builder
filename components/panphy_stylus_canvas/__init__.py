import streamlit.components.v1 as components
from pathlib import Path

# Declare the component. Streamlit will serve index.html from this folder.
_component = components.declare_component(
    name="panphy_stylus_canvas",
    path=str(Path(__file__).parent),
)

def stylus_canvas(
    *,
    width: int = 600,
    height: int = 520,
    stroke_width: int = 2,
    stroke_color: str = "#000000",
    background_color: str = "#F0F2F6",
    pen_only: bool = False,
    tool: str = "pen",
    command: str | None = None,
    command_nonce: int = 0,
    initial_data_url: str | None = None,
    key: str | None = None,
):
    """A lightweight stylus-friendly canvas component.

    - Width is responsive (fills Streamlit container). `width` is kept only as a legacy hint.
    - Height is the initial height; users can drag-resize the canvas vertically (like a textarea).
    - The component returns: {"data_url": "...", "is_empty": bool}
    """
    return _component(
        # args to JS
        canvas_width=int(width),
        canvas_height=int(height),
        stroke_width=int(stroke_width),
        stroke_color=str(stroke_color),
        background_color=str(background_color),
        pen_only=bool(pen_only),
        tool=str(tool),
        command=command,
        command_nonce=int(command_nonce),
        initial_data_url=initial_data_url,
        # iframe height (will be kept in sync by JS ResizeObserver)
        height=int(height),
        key=key,
        default=None,
    )
