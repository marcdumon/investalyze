"""How a plotly figure follows the app's light/dark switch.

One definition, so every page changes together when the switch does and a new figure has one thing
to call. Backgrounds are transparent: the card the figure sits in paints the surface, which is what
keeps a chart from showing as a dark block on a light page.
"""


def figure_layout(dark: bool) -> dict:
    """The layout keys that put a figure in the mode the switch is on."""
    return {'template': 'plotly_dark' if dark else 'plotly_white',
            'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)'}
