"""Source-name -> canonical PascalCase column map for the SimFin `companies` table.

SimFin's bulk CSV headers carry spaces/parentheses. The financial-statement tables
(income/balance/cashflow) keep them verbatim, but `companies` is metadata and is normalized
to the house PascalCase convention. Already-canonical cols (Ticker, SrcId, Src, Market, Industry,
Sector, IndustryId, ISIN, CIK) are not listed. Applied via DataFrame.rename in fundamental_data.py.
"""

COMPANIES = {
    'Company Name': 'CompanyName',
    'End of financial year (month)': 'FinancialYearEndMonth',
    'Number Employees': 'NumberEmployees',
    'Business Summary': 'BusinessSummary',
    'Main Currency': 'MainCurrency',
}
