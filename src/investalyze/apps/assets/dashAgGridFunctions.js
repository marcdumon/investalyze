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
