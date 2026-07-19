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


def _fence_range(values: pd.Series, own: float | None) -> tuple[float, float] | None:
    """Robust axis fences: 1.5 IQR beyond the quartiles, trimmed to the data and widened to include the ticker."""
    if values.empty:
        return None
    q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
    iqr = q3 - q1
    lo = max(float(values.min()), q1 - 1.5 * iqr)
    hi = min(float(values.max()), q3 + 1.5 * iqr)
    if own is not None and pd.notna(own):
        lo, hi = min(lo, float(own)), max(hi, float(own))
    if hi <= lo:
        return None
    return lo, hi


def _stacked_offsets(values: pd.Series) -> pd.Series:
    """Vertical offsets that stack pinned extremes in value order, highest at the top."""
    if len(values) == 1:
        return pd.Series([0.0], index=values.index)
    rank = values.rank(method='first')
    return -0.55 + (rank - 1) / (len(values) - 1) * 1.1


def _blend(start: tuple[int, int, int], end: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Linear RGB blend at t in [0, 1]."""
    return tuple(round(s + (e - s) * t) for s, e in zip(start, end))


# Status poles from the dataviz reference palette, validated for CVD separation and 3:1 contrast
# on both chart surfaces; the diverging midpoint is the neutral context gray.
_VERDICT_BAD = (208, 59, 59)     # #d03b3b
_VERDICT_GOOD = (12, 163, 12)    # #0ca30c
_VERDICT_GRAY = (137, 135, 129)  # #898781


def _verdict_rgb(score: float) -> tuple[int, int, int] | None:
    """RGB for an oriented 0-100 percentile in five verdict bins; the middle bin is neutral (None)."""
    if score >= 80:
        return _VERDICT_GOOD
    if score >= 60:
        return _blend(_VERDICT_GOOD, _VERDICT_GRAY, 0.5)
    if score > 40:
        return None
    if score > 20:
        return _blend(_VERDICT_BAD, _VERDICT_GRAY, 0.5)
    return _VERDICT_BAD


def peer_violin_figure(peers: pd.DataFrame, ticker: str, spec: list[tuple[str, str, str]], dark: bool) -> go.Figure:
    """Violin small multiples with inner quartile box and outlier dots; the ticker wears an accent needle plus diamond.

    Each oriented metric shows the ticker's peer percentile (P-tag, 100 = best per
    HIGHER_IS_BETTER) and tints the violin in one of five verdict bins from red (weak) through
    neutral gray to green (strong), so the number and the color always agree; metrics without an
    orientation stay gray, their P-tag a plain rank in the group. Dollar rows build the violin
    over log10 values so the shape reflects orders of magnitude, and label the axis in dollars.
    Other rows clamp the axis to robust IQR fences (widened to include the ticker) and pin values
    beyond them at the edge as outward triangles, so extremes stay visible without squeezing the
    body of the distribution.
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
                pct = pct if HIGHER_IS_BETTER[column] else 100 - pct
                verdict = _verdict_rgb(pct)
                if verdict is not None:
                    r, g, b = verdict
                    edge, fill = f'rgb({r}, {g}, {b})', f'rgba({r}, {g}, {b}, 0.3)'
        fences = _fence_range(x, own_x) if fmt != 'usd' and len(valid) else None
        inside = (x >= fences[0]) & (x <= fences[1]) if fences is not None else pd.Series(True, index=valid.index)
        if inside.any():
            fig.add_trace(go.Violin(
                x=x[inside], y0=0.0, orientation='h', width=1.6, spanmode='hard',
                points='outliers', jitter=0.25, pointpos=0, marker=point_style,
                box={'visible': True, 'width': 0.2, 'fillcolor': 'rgba(0, 0, 0, 0)',
                     'line': {'color': edge, 'width': 1}},
                text=valid['Ticker'][inside], customdata=customdata[inside] if customdata is not None else None,
                hoveron='points', hovertemplate=point_hover,
                line={'color': edge, 'width': 1}, fillcolor=fill,
                showlegend=False), row=i, col=1)
        if fences is not None and (~inside).any():
            extremes = x[~inside]
            sides = [extremes[extremes < fences[0]], extremes[extremes > fences[1]]]
            offsets = pd.concat([_stacked_offsets(side) for side in sides if len(side)]).reindex(extremes.index)
            fig.add_trace(go.Scatter(
                x=extremes.clip(fences[0], fences[1]), y=offsets,
                mode='markers', text=valid['Ticker'][~inside], customdata=extremes,
                marker={'size': 8, 'color': theme['context'], 'opacity': 0.9,
                        'symbol': ['triangle-right' if value > fences[1] else 'triangle-left' for value in extremes],
                        'line': {'width': 1, 'color': theme['ring']}},
                hovertemplate='%{text}: %{customdata:.2f} (beyond axis)<extra></extra>',
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
        else:
            if fences is not None:
                pad = (fences[1] - fences[0]) * 0.05
                fig.update_xaxes(range=[fences[0] - pad, fences[1] + pad], row=i, col=1)
            if fmt == 'pct':
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

    fig.add_trace(go.Scatter(x=dates, y=ttm['revenue'], line={'width': 2, 'color': slots[0]}, showlegend=False,
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


def item_history_figure(dates: pd.Series, values: pd.Series, label: str, dark: bool,
                        filed: tuple[pd.Series, pd.Series] | None = None) -> go.Figure:
    """One statement line item's full reported history as a single line, zero line visible.

    `filed` = (dates, values) overlays the as-filed series in context gray with a legend.
    """
    theme = _theme(dark)
    mode = 'lines+markers' if values.notna().sum() <= 40 else 'lines'
    fig = go.Figure()
    if filed is not None:
        fig.add_trace(go.Scatter(x=filed[0], y=filed[1], mode=mode, name='as filed',
                                 line={'width': 1.5, 'color': theme['context']}, marker={'size': 4},
                                 hovertemplate='%{y:,.3s}<extra>as filed</extra>'))
    fig.add_trace(go.Scatter(x=dates, y=values, mode=mode, name='restated', showlegend=filed is not None,
                             line={'width': 2, 'color': theme['accent']}, marker={'size': 5},
                             hovertemplate='%{y:,.3s}<extra>restated</extra>'))
    fig.update_layout(title={'text': label, 'font': {'size': 12, 'color': theme['context']}, 'x': 0},
                      legend={'orientation': 'h', 'x': 0, 'y': 1.2, 'font': {'size': 11}})
    fig.update_yaxes(tickformat='~s', zeroline=True)
    return _style(fig, dark, 240)
