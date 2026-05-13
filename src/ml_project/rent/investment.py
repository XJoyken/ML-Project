"""Investment math for rental property analysis.

All numbers are nominal KZT unless stated otherwise. We use a simple discounted
cash-flow model that treats one apartment unit as the investment vehicle:

  total_investment = sale_price + one_time_repair_cost
  annual_gross_rent = monthly_rent * 12 * (1 - vacancy_rate)
  annual_recurring_costs = maintenance + property_tax
  annual_amortised_costs = agent_commission_one_month / tenant_turnover_period
  annual_net_cashflow = annual_gross_rent - annual_recurring_costs - annual_amortised_costs

  gross_yield_pct = annual_gross_rent / total_investment * 100
  net_yield_pct = annual_net_cashflow / total_investment * 100
  payback_years_nominal = total_investment / annual_net_cashflow
  payback_years_inflation_adjusted: solve cumulative discounted cashflow >= total_investment
  npv_kzt at horizon_years: sum of discounted future cashflows minus initial investment
  irr_pct: discount rate that makes NPV == 0 (numerical solve)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from ..macro import InflationCatalog, RealEstateMarketCatalog

InvestmentVerdict = Literal["excellent", "good", "average", "poor"]


# -- Defaults ------------------------------------------------------------------
# Reasonable for the Almaty 2025 market based on industry rules of thumb.
DEFAULT_VACANCY_RATE = 0.08            # 8% of the year empty between tenants
DEFAULT_REPAIR_COST_PCT = 0.05         # 5% of sale price for initial repair/refurbish
DEFAULT_AGENT_COMMISSION_MONTHS = 0.5  # 0.5 month rent paid each turnover
DEFAULT_TENANT_TURNOVER_YEARS = 1.5    # avg tenant stays 1.5 years
DEFAULT_MAINTENANCE_PCT = 0.05         # 5% of annual gross rent for maintenance/repairs
DEFAULT_PROPERTY_TAX_PCT = 0.003       # 0.3% of sale price per year (Almaty 0.1-0.5% range)
DEFAULT_RISK_PREMIUM_PCT = 3.0         # 3 pp added to user-supplied inflation for discount rate
DEFAULT_HORIZON_YEARS = 10

# Payback thresholds for verdict (inflation-adjusted years).
VERDICT_PAYBACK_THRESHOLDS = {
    "excellent": 8.0,
    "good": 12.0,
    "average": 16.0,
}


@dataclass(slots=True)
class InvestmentParams:
    """User-overridable economic assumptions. Set any field to None to use default.

    The discount rate is NOT accepted directly from the user. Instead, supply:
      - inflation_rate_pct: annual CPI inflation (%). Defaults to the latest year from the
        macro dataset (2025 actual). The user sees and overrides this — it is intuitive.
      - risk_premium_pct: extra return required above inflation to justify illiquid real
        estate vs other investments. Defaults to 3 pp (conservative Almaty estimate).
    discount_rate_pct is computed as inflation_rate_pct + risk_premium_pct.
    """

    vacancy_rate: float | None = None
    repair_cost_pct: float | None = None
    agent_commission_months: float | None = None
    tenant_turnover_years: float | None = None
    maintenance_pct: float | None = None
    property_tax_pct: float | None = None
    inflation_rate_pct: float | None = None   # user-supplied; defaults to latest macro CPI
    risk_premium_pct: float | None = None     # added to inflation to get discount rate
    horizon_years: int | None = None

    def resolved(self, *, inflation: InflationCatalog) -> "ResolvedInvestmentParams":
        latest_pct = inflation.annual_pct(inflation.latest_year) or 0.0
        inflation_rate = _default(self.inflation_rate_pct, latest_pct)
        risk_premium = _default(self.risk_premium_pct, DEFAULT_RISK_PREMIUM_PCT)
        return ResolvedInvestmentParams(
            vacancy_rate=_default(self.vacancy_rate, DEFAULT_VACANCY_RATE),
            repair_cost_pct=_default(self.repair_cost_pct, DEFAULT_REPAIR_COST_PCT),
            agent_commission_months=_default(
                self.agent_commission_months, DEFAULT_AGENT_COMMISSION_MONTHS
            ),
            tenant_turnover_years=_default(
                self.tenant_turnover_years, DEFAULT_TENANT_TURNOVER_YEARS
            ),
            maintenance_pct=_default(self.maintenance_pct, DEFAULT_MAINTENANCE_PCT),
            property_tax_pct=_default(self.property_tax_pct, DEFAULT_PROPERTY_TAX_PCT),
            inflation_rate_pct=float(inflation_rate),
            risk_premium_pct=float(risk_premium),
            discount_rate_pct=float(inflation_rate + risk_premium),
            horizon_years=int(_default(self.horizon_years, DEFAULT_HORIZON_YEARS)),
        )


@dataclass(slots=True)
class ResolvedInvestmentParams:
    vacancy_rate: float
    repair_cost_pct: float
    agent_commission_months: float
    tenant_turnover_years: float
    maintenance_pct: float
    property_tax_pct: float
    inflation_rate_pct: float
    risk_premium_pct: float
    discount_rate_pct: float   # = inflation_rate_pct + risk_premium_pct
    horizon_years: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InvestmentResult:
    sale_price_kzt: float
    monthly_rent_kzt: float
    total_investment_kzt: float
    annual_gross_rent_kzt: float
    annual_recurring_costs_kzt: float
    annual_net_cashflow_kzt: float
    gross_yield_pct: float
    net_yield_pct: float
    payback_years_nominal: float | None
    payback_years_inflation_adjusted: float | None
    npv_kzt: float
    irr_pct: float | None
    market_yield_pct: float | None
    yield_vs_market_pct_points: float | None
    verdict: InvestmentVerdict
    params: ResolvedInvestmentParams

    def as_dict(self) -> dict[str, Any]:
        return {
            "sale_price_kzt": self.sale_price_kzt,
            "monthly_rent_kzt": self.monthly_rent_kzt,
            "total_investment_kzt": self.total_investment_kzt,
            "annual_gross_rent_kzt": self.annual_gross_rent_kzt,
            "annual_recurring_costs_kzt": self.annual_recurring_costs_kzt,
            "annual_net_cashflow_kzt": self.annual_net_cashflow_kzt,
            "gross_yield_pct": self.gross_yield_pct,
            "net_yield_pct": self.net_yield_pct,
            "payback_years_nominal": self.payback_years_nominal,
            "payback_years_inflation_adjusted": self.payback_years_inflation_adjusted,
            "npv_kzt": self.npv_kzt,
            "irr_pct": self.irr_pct,
            "market_yield_pct": self.market_yield_pct,
            "yield_vs_market_pct_points": self.yield_vs_market_pct_points,
            "verdict": self.verdict,
            "params": self.params.as_dict(),
        }


def compute_investment_metrics(
    *,
    sale_price_kzt: float,
    monthly_rent_kzt: float,
    params: InvestmentParams | None = None,
    inflation: InflationCatalog,
    market: RealEstateMarketCatalog,
) -> InvestmentResult:
    if sale_price_kzt <= 0:
        raise ValueError("sale_price_kzt must be positive")
    if monthly_rent_kzt <= 0:
        raise ValueError("monthly_rent_kzt must be positive")

    p = (params or InvestmentParams()).resolved(inflation=inflation)

    annual_gross_rent = monthly_rent_kzt * 12.0 * (1.0 - p.vacancy_rate)
    repair_one_time = sale_price_kzt * p.repair_cost_pct
    total_investment = sale_price_kzt + repair_one_time

    annual_maintenance = annual_gross_rent * p.maintenance_pct
    annual_property_tax = sale_price_kzt * p.property_tax_pct
    annual_agent_amortised = (
        monthly_rent_kzt * p.agent_commission_months / max(p.tenant_turnover_years, 0.1)
    )
    annual_recurring_costs = annual_maintenance + annual_property_tax + annual_agent_amortised

    annual_net_cashflow = annual_gross_rent - annual_recurring_costs

    gross_yield_pct = annual_gross_rent / total_investment * 100.0
    net_yield_pct = annual_net_cashflow / total_investment * 100.0
    payback_nominal = (
        total_investment / annual_net_cashflow if annual_net_cashflow > 0 else None
    )
    payback_real = _solve_payback_with_growth(
        total_investment=total_investment,
        first_year_cashflow=annual_net_cashflow,
        discount_rate=p.discount_rate_pct / 100.0,
        rent_growth_rate=max(market.average_growth("rent_kzt_m2_mo_thousand"), 0.0),
    )
    npv = _compute_npv(
        total_investment=total_investment,
        annual_cashflow=annual_net_cashflow,
        discount_rate=p.discount_rate_pct / 100.0,
        rent_growth_rate=max(market.average_growth("rent_kzt_m2_mo_thousand"), 0.0),
        horizon_years=p.horizon_years,
    )
    irr = _solve_irr(
        total_investment=total_investment,
        annual_cashflow=annual_net_cashflow,
        rent_growth_rate=max(market.average_growth("rent_kzt_m2_mo_thousand"), 0.0),
        horizon_years=p.horizon_years,
    )

    market_yield = market.latest().get("rental_yield_annual_pct")
    yield_vs_market = gross_yield_pct - market_yield if market_yield is not None else None

    verdict = _classify_verdict(
        payback_inflation_adjusted=payback_real,
        gross_yield_pct=gross_yield_pct,
        market_yield_pct=market_yield,
    )

    return InvestmentResult(
        sale_price_kzt=float(sale_price_kzt),
        monthly_rent_kzt=float(monthly_rent_kzt),
        total_investment_kzt=float(total_investment),
        annual_gross_rent_kzt=float(annual_gross_rent),
        annual_recurring_costs_kzt=float(annual_recurring_costs),
        annual_net_cashflow_kzt=float(annual_net_cashflow),
        gross_yield_pct=float(gross_yield_pct),
        net_yield_pct=float(net_yield_pct),
        payback_years_nominal=None if payback_nominal is None else float(payback_nominal),
        payback_years_inflation_adjusted=(
            None if payback_real is None else float(payback_real)
        ),
        npv_kzt=float(npv),
        irr_pct=None if irr is None else float(irr * 100.0),
        market_yield_pct=None if market_yield is None else float(market_yield),
        yield_vs_market_pct_points=(
            None if yield_vs_market is None else float(yield_vs_market)
        ),
        verdict=verdict,
        params=p,
    )


def _compute_npv(
    *,
    total_investment: float,
    annual_cashflow: float,
    discount_rate: float,
    rent_growth_rate: float,
    horizon_years: int,
) -> float:
    """NPV at the end of `horizon_years`, assuming rent grows at `rent_growth_rate`."""
    if discount_rate <= -1.0:
        return -total_investment
    npv = -total_investment
    for year in range(1, horizon_years + 1):
        cashflow = annual_cashflow * ((1.0 + rent_growth_rate) ** (year - 1))
        npv += cashflow / ((1.0 + discount_rate) ** year)
    return npv


def _solve_payback_with_growth(
    *,
    total_investment: float,
    first_year_cashflow: float,
    discount_rate: float,
    rent_growth_rate: float,
    max_years: int = 60,
) -> float | None:
    """Years until cumulative discounted cashflow recovers the investment."""
    if first_year_cashflow <= 0:
        return None
    cumulative = 0.0
    for year in range(1, max_years + 1):
        cashflow = first_year_cashflow * ((1.0 + rent_growth_rate) ** (year - 1))
        discounted = cashflow / ((1.0 + discount_rate) ** year)
        prev_cumulative = cumulative
        cumulative += discounted
        if cumulative >= total_investment:
            shortfall = total_investment - prev_cumulative
            return float(year - 1 + shortfall / discounted)
    return None


def _solve_irr(
    *,
    total_investment: float,
    annual_cashflow: float,
    rent_growth_rate: float,
    horizon_years: int,
    tol: float = 1e-5,
    max_iter: int = 200,
) -> float | None:
    """Bisection on the discount rate that drives NPV to zero. Returns rate as decimal."""
    if annual_cashflow <= 0:
        return None

    def npv_at(rate: float) -> float:
        if rate <= -1.0:
            return -total_investment
        s = -total_investment
        for year in range(1, horizon_years + 1):
            cashflow = annual_cashflow * ((1.0 + rent_growth_rate) ** (year - 1))
            s += cashflow / ((1.0 + rate) ** year)
        return s

    low, high = -0.99, 1.0
    npv_low, npv_high = npv_at(low), npv_at(high)
    # Bracket-expand until NPV changes sign
    expansion = 0
    while npv_low * npv_high > 0 and expansion < 40:
        if npv_high > 0:
            high = high * 2.0 + 0.5
        else:
            return None
        npv_high = npv_at(high)
        expansion += 1
    if npv_low * npv_high > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        npv_mid = npv_at(mid)
        if abs(npv_mid) < tol:
            return mid
        if npv_low * npv_mid < 0:
            high, npv_high = mid, npv_mid
        else:
            low, npv_low = mid, npv_mid
    return 0.5 * (low + high)


def _classify_verdict(
    *,
    payback_inflation_adjusted: float | None,
    gross_yield_pct: float,
    market_yield_pct: float | None,
) -> InvestmentVerdict:
    if payback_inflation_adjusted is None or math.isnan(payback_inflation_adjusted):
        # Cashflow never recovers investment within reasonable horizon.
        return "poor"
    base: InvestmentVerdict
    if payback_inflation_adjusted < VERDICT_PAYBACK_THRESHOLDS["excellent"]:
        base = "excellent"
    elif payback_inflation_adjusted < VERDICT_PAYBACK_THRESHOLDS["good"]:
        base = "good"
    elif payback_inflation_adjusted < VERDICT_PAYBACK_THRESHOLDS["average"]:
        base = "average"
    else:
        base = "poor"

    # Cross-check against market: if gross yield is significantly below market, downgrade once.
    if market_yield_pct is not None and gross_yield_pct < market_yield_pct * 0.7:
        order: list[InvestmentVerdict] = ["excellent", "good", "average", "poor"]
        idx = min(order.index(base) + 1, len(order) - 1)
        base = order[idx]
    return base


def _default(value: float | int | None, fallback: float) -> float:
    return float(fallback) if value is None else float(value)


__all__ = [
    "DEFAULT_VACANCY_RATE",
    "DEFAULT_REPAIR_COST_PCT",
    "DEFAULT_AGENT_COMMISSION_MONTHS",
    "DEFAULT_TENANT_TURNOVER_YEARS",
    "DEFAULT_MAINTENANCE_PCT",
    "DEFAULT_PROPERTY_TAX_PCT",
    "DEFAULT_RISK_PREMIUM_PCT",
    "DEFAULT_HORIZON_YEARS",
    "VERDICT_PAYBACK_THRESHOLDS",
    "InvestmentParams",
    "ResolvedInvestmentParams",
    "InvestmentResult",
    "InvestmentVerdict",
    "compute_investment_metrics",
]
