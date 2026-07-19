"""Tests for the ticker statements table: restatement dedupe, depth selection and view transforms."""

import numpy as np
import pandas as pd

from investalyze.apps.ticker import statements

META = {'Ticker': 'TST', 'SrcId': 1, 'Src': 's', 'Market': 'us', 'Period': 'A', 'IsRestated': True,
        'Currency': 'USD', 'Shares (Basic)': 1_000_000.0, 'Shares (Diluted)': 2_000_000.0}


def income_frame() -> pd.DataFrame:
    """Three annual periods with summary items, one detail item and one never-reported item."""
    rows = []
    for fy, revenue, net_income in ((2022, 100e6, 10e6), (2023, 120e6, -5e6), (2024, 150e6, 15e6)):
        rows.append({**META, 'Fiscal Year': fy, 'Fiscal Period': 'FY',
                     'Report Date': pd.Timestamp(f'{fy}-12-31'), 'Publish Date': pd.Timestamp(f'{fy + 1}-02-01'),
                     'Restated Date': pd.Timestamp(f'{fy + 1}-02-01'),
                     'Revenue': revenue, 'Cost of Revenue': -60e6, 'Gross Profit': revenue - 60e6,
                     'Foreign Exchange Gain (Loss)': 1e6, 'Interest Income': np.nan, 'Net Income': net_income})
    return pd.DataFrame(rows)


def by_item(rows: list[dict]) -> dict[str, dict]:
    return {row['item']: row for row in rows}


def test_latest_restated_keeps_last_restatement_per_period():
    frame = income_frame()
    older = frame.iloc[[0]].assign(**{'Restated Date': pd.Timestamp('2022-06-01'), 'Revenue': 999e6})
    deduped = statements.latest_restated(pd.concat([older, frame], ignore_index=True))
    assert len(deduped) == 3
    assert deduped.iloc[0]['Revenue'] == 100e6   # the newer restatement wins
    assert list(deduped['Fiscal Year']) == [2022, 2023, 2024]


def test_summary_depth_keeps_summary_items_and_levels():
    _, rows = statements.table_data(income_frame(), 'income', 'summary', 'usd')
    items = by_item(rows)
    assert 'Foreign Exchange Gain (Loss)' not in items
    assert items['Revenue']['level'] == 1
    assert items['Gross Profit']['level'] == 0


def test_detail_depth_adds_detail_items_and_drops_never_reported():
    _, rows = statements.table_data(income_frame(), 'income', 'detail', 'usd')
    items = by_item(rows)
    assert items['Foreign Exchange Gain (Loss)']['level'] == 2
    assert 'Interest Income' not in items   # all periods null


def test_usd_view_formats_millions():
    columns, rows = statements.table_data(income_frame(), 'income', 'summary', 'usd')
    items = by_item(rows)
    assert columns[0]['headerName'] == 'USD millions'
    assert items['Revenue']['FY 2024'] == '150'
    assert items['Net Income']['FY 2023'] == '-5.00'


def test_yoy_view_compares_same_period_prior_year():
    columns, rows = statements.table_data(income_frame(), 'income', 'summary', 'yoy')
    items = by_item(rows)
    assert columns[0]['headerName'] == 'YoY %'
    assert items['Revenue']['FY 2022'] == ''          # no prior year
    assert items['Revenue']['FY 2023'] == '+20.0%'
    assert items['Net Income']['FY 2024'] == '+400.0%'   # (15 - (-5)) / |-5|


def test_common_size_view_uses_revenue_for_income():
    _, rows = statements.table_data(income_frame(), 'income', 'summary', 'common')
    items = by_item(rows)
    assert items['Revenue']['FY 2024'] == '100.0%'
    assert items['Cost of Revenue']['FY 2024'] == '-40.0%'


def test_per_share_view_divides_by_diluted_shares():
    _, rows = statements.table_data(income_frame(), 'income', 'summary', 'ps')
    assert by_item(rows)['Net Income']['FY 2024'] == '7.50'


def test_columns_newest_first_and_last_window_trims():
    columns, rows = statements.table_data(income_frame(), 'income', 'summary', 'yoy', last=2)
    assert [column['field'] for column in columns[1:]] == ['FY 2024', 'FY 2023']
    assert by_item(rows)['Revenue']['FY 2023'] == '+20.0%'   # base year is outside the shown window


def test_diff_mode_shows_only_changed_cells():
    filed = income_frame().assign(IsRestated=False)
    filed.loc[filed['Fiscal Year'] == 2023, 'Revenue'] = 110e6   # restatement raised 2023 revenue by 10M
    columns, rows = statements.table_data(income_frame(), 'income', 'summary', 'usd', filed=filed)
    items = by_item(rows)
    assert columns[0]['headerName'] == 'diff USD millions'
    assert set(items) == {'Revenue'}                             # every other line is unchanged and dropped
    assert items['Revenue']['FY 2023'] == '+10.0'
    assert items['Revenue']['FY 2024'] == ''


def test_diff_mode_blank_when_filed_period_missing():
    filed = income_frame().assign(IsRestated=False)
    filed.loc[filed['Fiscal Year'] == 2024, 'Revenue'] = 140e6
    filed = filed[filed['Fiscal Year'] != 2022].reset_index(drop=True)
    _, rows = statements.table_data(income_frame(), 'income', 'summary', 'usd', filed=filed)
    revenue = by_item(rows)['Revenue']
    assert revenue['FY 2024'] == '+10.0'
    assert revenue['FY 2022'] == ''   # no as-filed row to compare against
