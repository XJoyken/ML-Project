from __future__ import annotations

import pytest

from ml_project.macro import load_inflation_catalog, load_market_catalog
from ml_project.rent.investment import (
    DEFAULT_HORIZON_YEARS,
    InvestmentParams,
    compute_investment_metrics,
)


@pytest.fixture(scope="module")
def macro():
    return load_inflation_catalog(), load_market_catalog()


def test_typical_apartment_yields_in_expected_band(macro):
    inflation, market = macro
    result = compute_investment_metrics(
        sale_price_kzt=45_000_000,
        monthly_rent_kzt=280_000,
        inflation=inflation,
        market=market,
    )
    # Gross yield for 45M / 280K monthly with 8% vacancy → ~6.5%
    assert 5.5 <= result.gross_yield_pct <= 7.5
    # Net yield is somewhat lower
    assert result.net_yield_pct < result.gross_yield_pct
    # Total investment includes 5% repair
    assert result.total_investment_kzt > 45_000_000
    # Verdict for ~17yr payback is "poor" by default thresholds
    assert result.verdict in {"average", "poor"}


def test_great_deal_gives_better_verdict(macro):
    inflation, market = macro
    # Same rent but half the sale price → much faster payback
    result = compute_investment_metrics(
        sale_price_kzt=20_000_000,
        monthly_rent_kzt=280_000,
        inflation=inflation,
        market=market,
    )
    assert result.gross_yield_pct > 14.0
    assert result.verdict in {"excellent", "good"}
    assert result.payback_years_inflation_adjusted is not None
    assert result.payback_years_inflation_adjusted < 12.0


def test_explicit_params_override_defaults(macro):
    inflation, market = macro
    params = InvestmentParams(
        vacancy_rate=0.0,
        repair_cost_pct=0.0,
        maintenance_pct=0.0,
        property_tax_pct=0.0,
        agent_commission_months=0.0,
        discount_rate_pct=0.0,
        horizon_years=10,
    )
    result = compute_investment_metrics(
        sale_price_kzt=12_000_000,
        monthly_rent_kzt=100_000,
        params=params,
        inflation=inflation,
        market=market,
    )
    # With no costs and 0% discount: annual gross = 1.2M, payback = 10 years exactly
    assert result.payback_years_nominal == pytest.approx(10.0, rel=1e-3)
    assert result.annual_recurring_costs_kzt == 0.0


def test_invalid_inputs_raise(macro):
    inflation, market = macro
    with pytest.raises(ValueError):
        compute_investment_metrics(
            sale_price_kzt=0,
            monthly_rent_kzt=100_000,
            inflation=inflation,
            market=market,
        )
    with pytest.raises(ValueError):
        compute_investment_metrics(
            sale_price_kzt=10_000_000,
            monthly_rent_kzt=-1,
            inflation=inflation,
            market=market,
        )


def test_result_dict_includes_all_keys(macro):
    inflation, market = macro
    result = compute_investment_metrics(
        sale_price_kzt=30_000_000,
        monthly_rent_kzt=200_000,
        inflation=inflation,
        market=market,
    ).as_dict()
    expected = {
        "sale_price_kzt",
        "monthly_rent_kzt",
        "total_investment_kzt",
        "annual_gross_rent_kzt",
        "annual_recurring_costs_kzt",
        "annual_net_cashflow_kzt",
        "gross_yield_pct",
        "net_yield_pct",
        "payback_years_nominal",
        "payback_years_inflation_adjusted",
        "npv_kzt",
        "irr_pct",
        "market_yield_pct",
        "yield_vs_market_pct_points",
        "verdict",
        "params",
    }
    assert expected.issubset(result.keys())
    assert result["params"]["horizon_years"] == DEFAULT_HORIZON_YEARS


def test_yield_vs_market_diff_sign(macro):
    inflation, market = macro
    # 12% yield should be ABOVE market 9.7%
    high = compute_investment_metrics(
        sale_price_kzt=24_000_000,
        monthly_rent_kzt=280_000,
        inflation=inflation,
        market=market,
    )
    assert high.yield_vs_market_pct_points is not None
    assert high.yield_vs_market_pct_points > 0
    # 4% yield should be BELOW market
    low = compute_investment_metrics(
        sale_price_kzt=60_000_000,
        monthly_rent_kzt=200_000,
        inflation=inflation,
        market=market,
    )
    assert low.yield_vs_market_pct_points is not None
    assert low.yield_vs_market_pct_points < 0
