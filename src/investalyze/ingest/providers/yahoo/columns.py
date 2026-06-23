"""Source-name -> canonical PascalCase column maps for the Yahoo metadata tables.

yfinance's `.info` keys are camelCase; we store them PascalCase to match the house schema
convention (Ticker, Src, Date, FetchedOn, ...). Already-canonical keys (Ticker, Src, FetchedOn)
are not listed. Applied via DataFrame.rename at write time in meta_data.py.
"""

COMPANY_PROFILE = {
    'address1': 'Address1', 'city': 'City', 'state': 'State', 'zip': 'Zip',
    'country': 'Country', 'website': 'Website', 'industry': 'Industry', 'sector': 'Sector',
    'longBusinessSummary': 'BusinessSummary', 'fullTimeEmployees': 'FullTimeEmployees',
    'auditRisk': 'AuditRisk', 'boardRisk': 'BoardRisk', 'compensationRisk': 'CompensationRisk',
    'shareHolderRightsRisk': 'ShareholderRightsRisk', 'overallRisk': 'OverallRisk',
    'irWebsite': 'IRWebsite',
}

COMPANY_OFFICERS = {
    'name': 'Name', 'title': 'Title', 'age': 'Age', 'yearBorn': 'YearBorn',
    'fiscalYear': 'FiscalYear', 'totalPay': 'TotalPay', 'exercisedValue': 'ExercisedValue',
    'unexercisedValue': 'UnexercisedValue',
}
