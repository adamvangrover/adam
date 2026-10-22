def calc_dscr(ebitda: float, capex: float, tax: float, debt_service: float) -> float:
    fcf = ebitda - capex - tax
    return round(fcf / debt_service, 4) if debt_service > 0 else float('inf')
