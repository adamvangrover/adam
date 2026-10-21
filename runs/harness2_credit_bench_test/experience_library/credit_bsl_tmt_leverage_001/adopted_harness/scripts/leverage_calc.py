def calc_net_leverage(total_debt: float, cash: float, adj_ebitda: float) -> float:
    net_debt = total_debt - cash
    return round(net_debt / adj_ebitda, 4) if adj_ebitda > 0 else float('inf')
