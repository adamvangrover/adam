from typing import Dict
import math


class ReturnMetrics:
    @staticmethod
    def calculate_moic(invested_capital: float, returned_capital: float) -> float:
        if invested_capital == 0:
            return 0.0
        return returned_capital / invested_capital

    @staticmethod
    def calculate_irr(invested_capital: float, returned_capital: float, years: float) -> float:
        """
        Calculates IRR for a single investment and exit event (CAGR).
        """
        if invested_capital == 0 or years == 0:
            return 0.0

        # If returned is 0, IRR is -100%
        if returned_capital <= 0:
            return -1.0

        moic = returned_capital / invested_capital

        # IRR = (MOIC ^ (1/T)) - 1
        return (moic ** (1.0 / years)) - 1
