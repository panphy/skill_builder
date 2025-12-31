import streamlit.components.v1 as components
from pathlib import Path

# Declare the component. Streamlit will serve index.html from this folder.
_component = components.declare_component(
    name="panphy_stylus_canvas",
    path=str(Path(__file__).parent),
)

def stylus_canvas(
    *,
    width: int,
    height: int,
    stroke_width: int = 2,
    stroke_color: str = "#000000",
    background_color: str = "#ffffff",
    pen_only: bool = False,
    tool: str = "pen",  # "pen" or "eraser"
    command: str | None = None,  # "undo" | "clear" | None
    command_nonce: int = 0,
    initial_data_url: str | None = None,
    key: str | None = None,
):
    """A lightweight drawing canvas with optional stylus-only mode (palm rejection).

    Returns a dict like:
      { "data_url": "data:image/png;base64,...", "is_empty": bool }
    """
    return _component(
        width=width,
        height=height,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=background_color,
        pen_only=pen_only,
        tool=tool,
        command=command,
        command_nonce=command_nonce,
        initial_data_url=initial_data_url,
        key=key,
        default=None,
    )
