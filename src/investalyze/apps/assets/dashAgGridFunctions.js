// Custom AG Grid cell renderers, looked up by name from columnDef.cellRenderer.
var dagcomponentfuncs = (window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {});

// Ticker cell as a real anchor: plain click opens the full analysis in a new focused tab,
// ctrl/middle click opens a background tab, exactly like any browser link.
dagcomponentfuncs.TickerLink = function (props) {
    return React.createElement(
        'a',
        {
            href: '/ticker?symbol=' + props.value,
            target: '_blank',
            rel: 'noopener',
            style: {color: 'var(--mantine-color-anchor)', textDecoration: 'none'},
        },
        props.value
    );
};

// Same link, but only when the ticker is actually a stock (IsStock flag set server-side from the
// screener pool); anomalies on bonds/indices/currencies or fundamentals-only tickers have nothing
// to show on the stock page, so those render as plain text.
dagcomponentfuncs.TickerLinkIfStock = function (props) {
    if (!props.data.IsStock) {
        return props.value;
    }
    return React.createElement(
        'a',
        {
            href: '/ticker?symbol=' + props.value,
            target: '_blank',
            rel: 'noopener',
            style: {color: 'var(--mantine-color-anchor)', textDecoration: 'none'},
        },
        props.value
    );
};
