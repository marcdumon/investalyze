"""Figure builders for the ticker analysis page.

Colors follow the dataviz reference palette: the subject ticker wears the accent hue, benchmark
and peers wear context gray, drawdown wears the red diverging pole, factor families wear the
five categorical slots in fixed order. Every builder takes prepared frames plus a dark flag and
returns a themed go.Figure with transparent backgrounds so it sits on the Mantine card.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from investalyze.analysis.factors import FAMILIES, HIGHER_IS_BETTER
from investalyze.apps.ticker.data import drawdown

_COLORS = {
    'light': {'accent': '#2a78d6', 'context': '#898781', 'neg': '#e34948', 'grid': '#e1e0d9',
              'axis': '#c3c2b7', 'ring': '#fcfcfb',
              'slots': ['#2a78d6', '#008300', '#e87ba4', '#eda100', '#1baf7a']},
    'dark': {'accent': '#3987e5', 'context': '#898781', 'neg': '#e66767', 'grid': '#2c2c2a',
             'axis': '#383835', 'ring': '#1a1a19',
             'slots': ['#3987e5', '#008300', '#d55181', '#c98500', '#199e70']},
}

# (pool column, row label, axis format) for the dot-strip sections
PEER_SPEC = [
    ('mcap', 'Market cap', 'usd'),
    ('earnings_yield', 'Earnings yield', 'pct'),
    ('sales_yield', 'Sales yield', 'pct'),
    ('fcf_yield', 'FCF yield', 'pct'),
    ('revenue_yoy', 'Revenue growth YoY', 'pct'),
    ('op_margin', 'Operating margin', 'pct'),
    ('roe', 'Return on equity', 'pct'),
]
RISK_SPEC = [
    ('vol_252', 'Volatility (1y, annualised)', 'pct'),
    ('beta_252', 'Beta vs S&P 500', 'x'),
    ('debt_to_equity', 'Debt / equity', 'x'),
]


def _theme(dark: bool) -> dict:
    """The color set for one mode."""
    return _COLORS['dark' if dark else 'light']


def _style(fig: go.Figure, dark: bool, height: int) -> go.Figure:
    """Shared chrome: template, transparent backgrounds, hairline grid, compact margins."""
    theme = _theme(dark)
    fig.update_layout(template='plotly_dark' if dark else 'plotly_white',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      height=height, margin={'l': 46, 'r': 60, 't': 34, 'b': 24},
                      font={'size': 12}, hoverlabel={'font': {'size': 11}})
    fig.update_xaxes(gridcolor=theme['grid'], linecolor=theme['axis'], zerolinecolor=theme['axis'])
    fig.update_yaxes(gridcolor=theme['grid'], linecolor=theme['axis'], zerolinecolor=theme['axis'])
    return fig


def _label_last(fig: go.Figure, x: pd.Series, y: pd.Series, text: str, color: str, row: int, col: int) -> None:
    """Direct label at a line's last non-NaN point."""
    valid = y.notna()
    if not valid.any():
        return
    last = valid[valid].index[-1]
    fig.add_annotation(x=x[last], y=y[last], text=text, font={'color': color, 'size': 11},
                       showarrow=False, xanchor='left', xshift=5, row=row, col=col)


def performance_figure(window: pd.DataFrame, ticker: str, dark: bool) -> go.Figure:
    """Indexed total return of the ticker vs the benchmark (log y) with the drawdown band beneath."""
    theme = _theme(dark)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=window['Date'], y=window['market'], name='S&P 500',
                             line={'width': 1.5, 'color': theme['context']},
                             hovertemplate='%{y:.0f}<extra>S&P 500</extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=window['Date'], y=window['AC'], name=ticker,
                             line={'width': 2, 'color': theme['accent']},
                             hovertemplate='%{y:.0f}<extra>' + ticker + '</extra>'), row=1, col=1)
    dd = drawdown(window['AC'].reset_index(drop=True))
    fig.add_trace(go.Scatter(x=window['Date'], y=dd, name='drawdown', fill='tozeroy',
                             line={'width': 1.5, 'color': theme['neg']}, showlegend=False,
                             hovertemplate='%{y:.0%}<extra>drawdown</extra>'), row=2, col=1)
    fig.update_yaxes(type='log', row=1, col=1, title={'text': 'indexed = 100', 'font': {'size': 11}})
    fig.update_yaxes(tickformat='.0%', row=2, col=1)
    fig.update_layout(hovermode='x unified',
                      legend={'orientation': 'h', 'x': 0, 'y': 1.08, 'font': {'size': 11}})
    return _style(fig, dark, 440)


def trailing_returns_figure(returns: pd.DataFrame, ticker: str, dark: bool) -> go.Figure:
    """Horizontal grouped bars: ticker vs benchmark total return per trailing window."""
    theme = _theme(dark)
    ordered = returns.iloc[::-1]   # longest window at the bottom, 1m on top
    fig = go.Figure()
    fig.add_trace(go.Bar(y=ordered['window'], x=ordered['market'], orientation='h', name='S&P 500',
                         marker={'color': theme['context']}, hovertemplate='%{x:.1%}<extra>S&P 500</extra>'))
    fig.add_trace(go.Bar(y=ordered['window'], x=ordered['ticker'], orientation='h', name=ticker,
                         marker={'color': theme['accent']}, hovertemplate='%{x:.1%}<extra>' + ticker + '</extra>'))
    fig.update_xaxes(tickformat='.0%')
    fig.update_layout(barmode='group', bargap=0.35,
                      legend={'orientation': 'h', 'x': 0, 'y': 1.12, 'font': {'size': 11}})
    return _style(fig, dark, 300)


def factor_profile_figure(families: pd.Series, ranks: pd.Series, dark: bool) -> go.Figure:
    """Percentile-vs-peers bars: one per factor family on top, one per factor beneath.

    Bars wear their family's categorical slot; the solid reference line marks the peer median (50).
    """
    theme = _theme(dark)
    family_colors = {family: theme['slots'][i] for i, family in enumerate(FAMILIES)}
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.24, 0.76], vertical_spacing=0.06,
                        subplot_titles=('family score', 'individual factors'))
    fig.add_trace(go.Bar(
        y=list(reversed(families.index.tolist())), x=list(reversed(families.tolist())), orientation='h',
        marker={'color': [family_colors[f] for f in reversed(families.index.tolist())]},
        hovertemplate='%{x:.0f} pctile<extra>%{y}</extra>', showlegend=False), row=1, col=1)
    factor_names, factor_colors = [], []
    for family, members in FAMILIES.items():
        for factor in members:
            factor_names.append(factor)
            factor_colors.append(family_colors[family])
    factor_names.reverse()
    factor_colors.reverse()
    fig.add_trace(go.Bar(
        y=factor_names, x=[ranks[name] for name in factor_names], orientation='h',
        marker={'color': factor_colors},
        hovertemplate='%{x:.0f} pctile<extra>%{y}</extra>', showlegend=False), row=2, col=1)
    for row in (1, 2):
        fig.add_vline(x=50, line_width=1, line_color=theme['context'], row=row, col=1)
    fig.update_xaxes(range=[0, 103], tickvals=[0, 25, 50, 75, 100], row=2, col=1)
    fig.update_layout(bargap=0.35)
    for annotation in fig.layout.annotations:
        annotation.update(x=0, xanchor='left', font={'size': 11, 'color': theme['context']})
    return _style(fig, dark, 620)


def _usd_tick(exponent: int) -> str:
    """Dollar tick label for 10**exponent, matching plotly's ~s suffixes."""
    unit = {1: 'k', 2: 'M', 3: 'G', 4: 'T'}.get(exponent // 3, '')
    return f'${10 ** (exponent % 3)}{unit}'


def _blend(start: tuple[int, int, int], end: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Linear RGB blend at t in [0, 1]."""
    return tuple(round(s + (e - s) * t) for s, e in zip(start, end))


def _verdict_rgb(score: float) -> tuple[int, int, int]:
    """Red through gray to green RGB for a 0-100 goodness score."""
    red, gray, green = (227, 73, 72), (137, 135, 129), (25, 158, 112)
    if score <= 50:
        return _blend(red, gray, score / 50)
    return _blend(gray, green, (score - 50) / 50)


def peer_violin_figure(peers: pd.DataFrame, ticker: str, spec: list[tuple[str, str, str]], dark: bool) -> go.Figure:
    """Violin small multiples with inner quartile box and outlier dots; the ticker wears an accent needle plus diamond.

    Each violin is tinted red through gray to green by the ticker's oriented percentile in that
    metric (HIGHER_IS_BETTER decides the direction; unoriented metrics stay gray), with a P-tag
    showing the raw percentile. Dollar rows build the violin over log10 values so the shape
    reflects orders of magnitude, and label the axis in dollars. Axes span the full data range
    so outliers stay visible.
    """
    theme = _theme(dark)
    fig = make_subplots(rows=len(spec), cols=1, vertical_spacing=min(0.5 / max(len(spec) - 1, 1), 0.14),
                        subplot_titles=[label for _, label, _ in spec])
    for annotation in fig.layout.annotations:   # only the subplot titles exist at this point
        annotation.update(x=0, xanchor='left', font={'size': 11, 'color': theme['context']})
    others = peers[peers['Ticker'] != ticker]
    me = peers[peers['Ticker'] == ticker]
    point_style = {'size': 7, 'color': theme['context'], 'opacity': 0.8, 'line': {'width': 1, 'color': theme['ring']}}
    for i, (column, _label, fmt) in enumerate(spec, start=1):
        own = float(me.iloc[0][column]) if len(me) and pd.notna(me.iloc[0][column]) else None
        if fmt == 'usd':
            valid = others[others[column].notna() & (others[column] > 0)]
            x = np.log10(valid[column])
            own_x = np.log10(own) if own is not None and own > 0 else None
            customdata = valid[column]
            point_hover = '%{text}: %{customdata:,.3s}<extra></extra>'
            own_hover = '%{text}: %{customdata:,.3s}<extra></extra>'
        else:
            valid = others[others[column].notna()]
            x = valid[column]
            own_x = own
            customdata = None
            point_hover = '%{text}: %{x:.2f}<extra></extra>'
            own_hover = '%{text}: %{x:.2f}<extra></extra>'
        pct, edge, fill = None, theme['context'], 'rgba(137, 135, 129, 0.35)'
        if own is not None and len(valid):
            pct = float((valid[column] <= own).mean() * 100)
            if column in HIGHER_IS_BETTER:
                r, g, b = _verdict_rgb(pct if HIGHER_IS_BETTER[column] else 100 - pct)
                edge, fill = f'rgb({r}, {g}, {b})', f'rgba({r}, {g}, {b}, 0.3)'
        if len(valid):
            fig.add_trace(go.Violin(
                x=x, y0=0.0, orientation='h', width=1.6, spanmode='hard',
                points='outliers', jitter=0.25, pointpos=0, marker=point_style,
                box={'visible': True, 'width': 0.2, 'fillcolor': 'rgba(0, 0, 0, 0)',
                     'line': {'color': edge, 'width': 1}},
                text=valid['Ticker'], customdata=customdata, hoveron='points', hovertemplate=point_hover,
                line={'color': edge, 'width': 1}, fillcolor=fill,
                showlegend=False), row=i, col=1)
        if pct is not None:
            fig.add_annotation(text=f'P{pct:.0f}', x=1, y=1, xref='x domain', yref='y domain',
                               xanchor='right', yanchor='top', showarrow=False,
                               font={'size': 11, 'color': edge}, row=i, col=1)
        if own_x is not None:
            fig.add_trace(go.Scatter(
                x=[own_x, own_x], y=[-0.8, 0.8], mode='lines', line={'color': theme['accent'], 'width': 2},
                hoverinfo='skip', showlegend=False), row=i, col=1)
            fig.add_trace(go.Scatter(
                x=[own_x], y=[0], mode='markers', text=[ticker], customdata=[own], marker={
                    'size': 14, 'symbol': 'diamond', 'color': theme['accent'],
                    'line': {'width': 2, 'color': theme['ring']}},
                hovertemplate=own_hover, showlegend=False), row=i, col=1)
        fig.update_yaxes(visible=False, range=[-1, 1], row=i, col=1)
        if fmt == 'usd':
            exponents = list(x) + ([own_x] if own_x is not None else [])
            if exponents:
                lo, hi = int(np.floor(min(exponents))), int(np.ceil(max(exponents)))
                fig.update_xaxes(tickvals=list(range(lo, hi + 1)),
                                 ticktext=[_usd_tick(k) for k in range(lo, hi + 1)], row=i, col=1)
        elif fmt == 'pct':
            fig.update_xaxes(tickformat='.0%', row=i, col=1)
    return _style(fig, dark, 108 * len(spec) + 40)


def fundamentals_figure(ttm: pd.DataFrame, dark: bool) -> go.Figure:
    """Small multiples over the quarterly TTM history: growth, margins, per-share, balance items."""
    theme = _theme(dark)
    slots = theme['slots']
    dates = ttm['Report Date']
    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.16, horizontal_spacing=0.07,
                        subplot_titles=('Revenue TTM', 'Margins TTM', 'Per share TTM',
                                        'Debt vs equity', 'Cash & ST investments', 'Diluted shares'))
    for annotation in fig.layout.annotations:   # only the subplot titles exist at this point
        annotation.update(font={'size': 12, 'color': theme['context']})

    fig.add_trace(go.Scatter(x=dates, y=ttm['revenue'], line={'width': 2, 'color': slots[0]},
                             fill='tozeroy', showlegend=False,
                             hovertemplate='%{y:,.3s}<extra>revenue</extra>'), row=1, col=1)

    margin_series = [('gross_margin', 'gross', slots[0]), ('op_margin', 'op', slots[1]), ('net_margin', 'net', slots[2])]
    for column, label, color in margin_series:
        fig.add_trace(go.Scatter(x=dates, y=ttm[column], line={'width': 2, 'color': color}, showlegend=False,
                                 hovertemplate='%{y:.1%}<extra>' + label + '</extra>'), row=1, col=2)
        _label_last(fig, dates, ttm[column], label, color, row=1, col=2)

    per_share = [('eps', 'EPS', slots[0]), ('fcf_ps', 'FCF', slots[1])]
    for column, label, color in per_share:
        fig.add_trace(go.Scatter(x=dates, y=ttm[column], line={'width': 2, 'color': color}, showlegend=False,
                                 hovertemplate='%{y:.2f}<extra>' + label + '</extra>'), row=1, col=3)
        _label_last(fig, dates, ttm[column], label, color, row=1, col=3)

    balance_series = [('equity', 'equity', slots[0]), ('debt', 'debt', slots[1])]
    for column, label, color in balance_series:
        fig.add_trace(go.Scatter(x=dates, y=ttm[column], line={'width': 2, 'color': color}, showlegend=False,
                                 hovertemplate='%{y:,.3s}<extra>' + label + '</extra>'), row=2, col=1)
        _label_last(fig, dates, ttm[column], label, color, row=2, col=1)

    fig.add_trace(go.Scatter(x=dates, y=ttm['cash'], line={'width': 2, 'color': slots[0]}, showlegend=False,
                             hovertemplate='%{y:,.3s}<extra>cash</extra>'), row=2, col=2)
    fig.add_trace(go.Scatter(x=dates, y=ttm['shares'], line={'width': 2, 'color': slots[0]}, showlegend=False,
                             hovertemplate='%{y:,.3s}<extra>shares</extra>'), row=2, col=3)

    fig.update_yaxes(tickformat='~s', row=1, col=1)
    fig.update_yaxes(tickformat='.0%', row=1, col=2)
    fig.update_yaxes(tickformat='~s', row=2, col=1)
    fig.update_yaxes(tickformat='~s', row=2, col=2)
    fig.update_yaxes(tickformat='~s', row=2, col=3)
    return _style(fig, dark, 520)
