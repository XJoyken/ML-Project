"use client";

import { useState } from "react";
import { useLang } from "@/context/LanguageContext";
import { t } from "@/lib/i18n";
import { investment, ApiUnavailable, ApiBadRequest } from "@/lib/api";
import type { InvestmentResponse, InvestmentParams } from "@/lib/types";
import BackButton from "@/components/BackButton";
import VerdictBadge from "@/components/VerdictBadge";
import LoadingOverlay from "@/components/LoadingOverlay";
import FreshnessWarning from "@/components/FreshnessWarning";
import Tooltip from "@/components/Tooltip";

// ── Format helpers ────────────────────────────────────────────
function fmtPrice(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)} млн ₸`;
  return `${(n / 1_000).toFixed(0)} тыс ₸`;
}
function fmtPct(n: number | null | undefined): string {
  if (n == null) return "—";
  return `${n.toFixed(2)}%`;
}
function fmtYears(n: number | null | undefined): string {
  if (n == null) return "—";
  return `${n.toFixed(1)} лет`;
}
function fmtNpv(n: number | null | undefined): string {
  if (n == null) return "—";
  const sign = n >= 0 ? "+" : "";
  if (Math.abs(n) >= 1_000_000) return `${sign}${(n / 1_000_000).toFixed(1)} млн ₸`;
  return `${sign}${(n / 1_000).toFixed(0)} тыс ₸`;
}
function isKrishaUrl(url: string): boolean {
  return /krisha\.kz\/a\/show\/\d+/.test(url);
}

function ScenarioCard({
  lang,
  variant,
  title,
  subtitle,
  rent,
  metrics,
}: {
  lang: "ru" | "en";
  variant: "conservative" | "base" | "optimistic";
  title: string;
  subtitle: string;
  rent: number | null | undefined;
  metrics: InvestmentResponse["investment"] | null | undefined;
}) {
  const verdict = metrics?.verdict ?? "average";
  return (
    <div className={`scenario-card scenario-card--${variant}`}>
      <div className="scenario-card-head">
        <div className="scenario-card-title">{title}</div>
        <div className="scenario-card-sub">{subtitle}</div>
      </div>
      <div className="scenario-rent">{fmtPrice(rent)}<span style={{ color: "var(--text-muted)", fontSize: "0.78rem", fontWeight: 400 }}> /{lang === "ru" ? "мес" : "mo"}</span></div>
      {metrics && <VerdictBadge verdict={verdict} />}
      <div className="scenario-stats">
        <div className="scenario-stat">
          <span>{t(lang, "inv.s.net_yield")}</span>
          <strong>{fmtPct(metrics?.net_yield_pct)}</strong>
        </div>
        <div className="scenario-stat">
          <span>{t(lang, "inv.s.payback")}</span>
          <strong>{fmtYears(metrics?.payback_years_inflation_adjusted)}</strong>
        </div>
        <div className="scenario-stat">
          <span>{t(lang, "inv.s.npv")}</span>
          <strong style={{ color: (metrics?.npv_kzt ?? 0) >= 0 ? "var(--success)" : "var(--danger)" }}>
            {fmtNpv(metrics?.npv_kzt)}
          </strong>
        </div>
      </div>
    </div>
  );
}

function ThresholdRow({
  active, color, label, lang,
}: { active: boolean; color: string; label: string; lang: "ru" | "en" }) {
  return (
    <li
      style={{
        display: "flex",
        alignItems: "center",
        gap: "0.6rem",
        padding: "0.4rem 0.6rem",
        borderRadius: "8px",
        background: active ? `${color}22` : "transparent",
        border: active ? `1px solid ${color}66` : "1px solid transparent",
        fontWeight: active ? 600 : 400,
      }}
    >
      <span
        style={{
          width: 10, height: 10, borderRadius: "50%",
          background: color, flexShrink: 0,
          boxShadow: active ? `0 0 8px ${color}` : "none",
        }}
      />
      {label}
      {active && (
        <span style={{ marginLeft: "auto", fontSize: "0.78rem", color }}>
          ← {lang === "ru" ? "ваш вердикт" : "your verdict"}
        </span>
      )}
    </li>
  );
}

// ── Field config ─────────────────────────────────────────────
// All fields are pre-filled with defaults. User sees real values, edits them.
// Format hint (under field) = how to enter (e.g. "0.003 = 0.3%").
// Tooltip (on ?) = full explanation of what the field means.

const DEFAULTS = {
  vacancy_rate: 0.08,
  repair_cost_pct: 0.05,
  agent_commission_months: 0.5,
  tenant_turnover_years: 1.5,
  maintenance_pct: 0.05,
  property_tax_pct: 0.003,
  inflation_rate_pct: 12.3,
  risk_premium_pct: 3.0,
  horizon_years: 10,
} as const;

interface FieldSpec {
  id: keyof typeof DEFAULTS;
  labelKey: string;
  tooltipKey: string;
  hint: { ru: string; en: string };
  step: number;
  min: number;
  max: number;
}

const FIELDS: FieldSpec[] = [
  {
    id: "vacancy_rate",
    labelKey: "inv.vacancy",
    tooltipKey: "inv.vacancy_help",
    hint: { ru: "доля от года (0.08 = 8%)", en: "share of year (0.08 = 8%)" },
    step: 0.01, min: 0, max: 0.9,
  },
  {
    id: "repair_cost_pct",
    labelKey: "inv.repair",
    tooltipKey: "inv.repair_help",
    hint: { ru: "доля от цены (0.05 = 5%)", en: "share of price (0.05 = 5%)" },
    step: 0.01, min: 0, max: 0.5,
  },
  {
    id: "agent_commission_months",
    labelKey: "inv.agent",
    tooltipKey: "inv.agent_help",
    hint: { ru: "в месяцах аренды (0.5 = пол месяца)", en: "in months of rent (0.5 = half a month)" },
    step: 0.1, min: 0, max: 3,
  },
  {
    id: "tenant_turnover_years",
    labelKey: "inv.turnover",
    tooltipKey: "inv.turnover_help",
    hint: { ru: "в годах (1.5 = раз в 1.5 года)", en: "in years (1.5 = once in 1.5 yrs)" },
    step: 0.5, min: 0.5, max: 10,
  },
  {
    id: "maintenance_pct",
    labelKey: "inv.maintenance",
    tooltipKey: "inv.maintenance_help",
    hint: { ru: "доля от годовой аренды (0.05 = 5%)", en: "share of annual rent (0.05 = 5%)" },
    step: 0.01, min: 0, max: 0.3,
  },
  {
    id: "property_tax_pct",
    labelKey: "inv.tax",
    tooltipKey: "inv.tax_help",
    hint: { ru: "доля от цены в год (0.003 = 0.3%)", en: "share of price per year (0.003 = 0.3%)" },
    step: 0.001, min: 0, max: 0.05,
  },
  {
    id: "inflation_rate_pct",
    labelKey: "inv.inflation",
    tooltipKey: "inv.inflation_help",
    hint: { ru: "в процентах (12.3 = 12.3%)", en: "in percent (12.3 = 12.3%)" },
    step: 0.1, min: 0, max: 100,
  },
  {
    id: "risk_premium_pct",
    labelKey: "inv.risk",
    tooltipKey: "inv.risk_help",
    hint: { ru: "в процентных пунктах (3 = 3 пп)", en: "in percentage points (3 = 3 pp)" },
    step: 0.5, min: 0, max: 20,
  },
  {
    id: "horizon_years",
    labelKey: "inv.horizon",
    tooltipKey: "inv.horizon_help",
    hint: { ru: "целое число лет (10 = 10 лет)", en: "integer years (10 = 10 years)" },
    step: 1, min: 1, max: 30,
  },
];

export default function InvestmentPage() {
  const { lang } = useLang();
  const [url, setUrl] = useState("");
  const [useLlm, setUseLlm] = useState(true);
  const [showParams, setShowParams] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InvestmentResponse | null>(null);

  // Pre-filled with defaults
  const [params, setParams] = useState<Record<string, number>>({ ...DEFAULTS });

  function updateField(key: string, raw: string) {
    const v = parseFloat(raw);
    if (isNaN(v)) return;
    setParams((p) => ({ ...p, [key]: v }));
  }

  function validateParams(): string | null {
    for (const f of FIELDS) {
      const v = params[f.id];
      if (v == null || isNaN(v)) {
        return `${t(lang, f.labelKey)}: ${lang === "ru" ? "введите число" : "enter a number"}`;
      }
      if (v < f.min || v > f.max) {
        return `${t(lang, f.labelKey)}: ${lang === "ru" ? "вне диапазона" : "out of range"} (${f.min}–${f.max})`;
      }
    }
    return null;
  }

  function buildPayload(): InvestmentParams {
    return {
      vacancy_rate:            params.vacancy_rate,
      repair_cost_pct:         params.repair_cost_pct,
      agent_commission_months: params.agent_commission_months,
      tenant_turnover_years:   params.tenant_turnover_years,
      maintenance_pct:         params.maintenance_pct,
      property_tax_pct:        params.property_tax_pct,
      inflation_rate_pct:      params.inflation_rate_pct,
      risk_premium_pct:        params.risk_premium_pct,
      horizon_years:           Math.round(params.horizon_years),
    };
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);

    if (!url.trim()) { setError(t(lang, "err.url_required")); return; }
    if (!isKrishaUrl(url)) { setError(t(lang, "err.url_invalid")); return; }
    const validationError = validateParams();
    if (validationError) { setError(validationError); return; }

    setLoading(true);
    try {
      const data = await investment({
        url,
        language: lang,
        use_llm: useLlm,
        investment_params: buildPayload(),
      });
      setResult(data);
    } catch (err) {
      if (err instanceof ApiUnavailable) setError(t(lang, "err.unavailable"));
      else if (err instanceof ApiBadRequest)
        setError(t(lang, "err.bad_request", { detail: (err as Error).message }));
      else setError(t(lang, "err.unknown", { detail: String((err as Error).message) }));
    } finally {
      setLoading(false);
    }
  }

  const narr = result?.narrative;
  const inv  = result?.investment;

  return (
    <div className="page-col fade-up">
      <BackButton />
      <h1 style={{ fontSize: "1.7rem", fontWeight: 800, marginBottom: "0.3rem" }}>
        {t(lang, "inv.title")}
      </h1>
      <p style={{ color: "var(--text-muted)", marginBottom: "1.8rem", fontSize: "0.95rem" }}>
        {lang === "ru"
          ? "Доходность, окупаемость, NPV/IRR по квартире с krisha.kz"
          : "Yield, payback, NPV/IRR for any krisha.kz listing"}
      </p>

      <FreshnessWarning />

      <form onSubmit={handleSubmit} style={{ marginBottom: "1.5rem" }}>
        {/* URL */}
        <div style={{ marginBottom: "1rem" }}>
          <label className="input-label" htmlFor="inv-url">{t(lang, "inv.url_label")}</label>
          <input
            id="inv-url"
            className="input-field"
            type="url"
            placeholder={t(lang, "inv.url_placeholder")}
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            autoComplete="off"
          />
        </div>

        {/* LLM toggle */}
        <div className="toggle-row" style={{ marginBottom: "1rem" }}>
          <label className="toggle-switch" htmlFor="inv-llm">
            <input id="inv-llm" type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
            <div className="toggle-track" />
          </label>
          <div className="toggle-label-group">
            <span className="toggle-label">{t(lang, "inv.use_llm")}</span>
            <span className="toggle-help">{t(lang, "inv.use_llm_help")}</span>
          </div>
        </div>

        {/* Investment params expander */}
        <div className="expander" style={{ marginBottom: "1.2rem" }}>
          <button
            type="button"
            className="expander-header"
            onClick={() => setShowParams(!showParams)}
          >
            <span>⚙ {t(lang, "inv.params")}</span>
            <span style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
              {showParams ? "▲" : "▼"}
            </span>
          </button>
          {showParams && (
            <div className="expander-body">
              <div className="param-grid">
                {FIELDS.map((f) => (
                  <div key={f.id}>
                    <label className="input-label" htmlFor={`inv-${f.id}`} style={{ display: "flex", alignItems: "center" }}>
                      {t(lang, f.labelKey)}
                      <Tooltip text={t(lang, f.tooltipKey)} />
                    </label>
                    <input
                      id={`inv-${f.id}`}
                      className="input-field"
                      type="number"
                      step={f.step}
                      min={f.min}
                      max={f.max}
                      value={params[f.id]}
                      onChange={(e) => updateField(f.id, e.target.value)}
                    />
                    <div className="input-hint">{f.hint[lang]}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? "..." : t(lang, "inv.submit")}
        </button>
      </form>

      {/* Error */}
      {error && <div className="callout-danger" style={{ marginBottom: "1rem" }}>{error}</div>}

      {/* Loading */}
      {loading && <LoadingOverlay />}

      {/* Fallback notice */}
      {!loading && narr?.source === "fallback" && narr.fallback_reason && (
        <div className="callout-amber" style={{ marginBottom: "1rem" }}>
          {t(lang, "err.fallback_notice")}{narr.fallback_reason}
        </div>
      )}

      {/* Results */}
      {!loading && narr && inv && result && (
        <div className="fade-up">
          {/* Verdict */}
          <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1.4rem" }}>
            <VerdictBadge verdict={narr.verdict ?? inv.verdict} />
          </div>

          {/* Verdict thresholds explainer */}
          <details className="expander" style={{ marginBottom: "1.4rem" }}>
            <summary className="expander-header">
              <span>📊 {t(lang, "inv.r.thresholds_title")}</span>
              <span style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>▼</span>
            </summary>
            <div className="expander-body" style={{ fontSize: "0.9rem", lineHeight: 1.6 }}>
              <div style={{ marginBottom: "0.7rem", color: "var(--text-muted)" }}>
                {t(lang, "inv.r.thresholds_intro")}
              </div>
              <ul style={{ listStyle: "none", padding: 0, margin: "0 0 0.9rem 0" }}>
                <ThresholdRow lang={lang} active={(narr.verdict ?? inv.verdict) === "excellent"} color="var(--success)" label={t(lang, "inv.r.threshold_excellent")} />
                <ThresholdRow lang={lang} active={(narr.verdict ?? inv.verdict) === "good"}      color="#84cc16"          label={t(lang, "inv.r.threshold_good")} />
                <ThresholdRow lang={lang} active={(narr.verdict ?? inv.verdict) === "average"}   color="var(--warning)"   label={t(lang, "inv.r.threshold_average")} />
                <ThresholdRow lang={lang} active={(narr.verdict ?? inv.verdict) === "poor"}      color="var(--danger)"    label={t(lang, "inv.r.threshold_poor")} />
              </ul>
              <div style={{ marginBottom: "0.7rem" }}>
                <strong>{t(lang, "inv.r.threshold_current")}</strong>{" "}
                <span style={{ color: "var(--accent)", fontWeight: 700 }}>
                  {inv.payback_years_inflation_adjusted == null
                    ? "—"
                    : `${inv.payback_years_inflation_adjusted.toFixed(1)} ${lang === "ru" ? "лет" : "yr"}`}
                </span>
              </div>
              <div style={{ color: "var(--text-muted)" }}>
                {t(lang, "inv.r.thresholds_context")}
              </div>
            </div>
          </details>

          {/* Investment stats */}
          <div className="stat-grid">
            <div className="stat-card">
              <div className="stat-label">{lang === "ru" ? "Валовая доходность" : "Gross yield"}</div>
              <div className="stat-value">{fmtPct(inv.gross_yield_pct)}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">{lang === "ru" ? "Чистая доходность" : "Net yield"}</div>
              <div className="stat-value">{fmtPct(inv.net_yield_pct)}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">{lang === "ru" ? "Окупаемость (номин.)" : "Payback (nominal)"}</div>
              <div className="stat-value">{fmtYears(inv.payback_years_nominal)}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">{lang === "ru" ? "Окупаемость (с инфляцией)" : "Payback (real)"}</div>
              <div className="stat-value">{fmtYears(inv.payback_years_inflation_adjusted)}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">NPV</div>
              <div
                className="stat-value"
                style={{ color: inv.npv_kzt >= 0 ? "var(--success)" : "var(--danger)" }}
              >
                {fmtNpv(inv.npv_kzt)}
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">IRR</div>
              <div className="stat-value">{fmtPct(inv.irr_pct)}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">{lang === "ru" ? "Цена квартиры" : "Sale price"}</div>
              <div className="stat-value">{fmtPrice(result.sale_evaluation.listing_price_kzt)}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">{lang === "ru" ? "Прогноз аренды/мес" : "Predicted rent/mo"}</div>
              <div className="stat-value">{fmtPrice(result.rent_evaluation.predicted_rent_kzt)}</div>
            </div>
          </div>

          {/* Rent-scenarios sensitivity panel */}
          {result.rent_evaluation.rent_interval && (
            <>
              <div className="section-title">📊 {t(lang, "inv.scenarios_title")}</div>
              <div style={{ color: "var(--text-muted)", fontSize: "0.9rem", lineHeight: 1.55, marginBottom: "0.6rem" }}>
                {t(lang, "inv.scenarios_lead")}
              </div>
              <div className="scenarios-grid">
                <ScenarioCard
                  lang={lang}
                  variant="conservative"
                  title={t(lang, "inv.scenario_conservative")}
                  subtitle={t(lang, "inv.scenario_conservative_sub")}
                  rent={result.rent_evaluation.rent_interval.low_kzt}
                  metrics={result.investment_conservative ?? null}
                />
                <ScenarioCard
                  lang={lang}
                  variant="base"
                  title={t(lang, "inv.scenario_base")}
                  subtitle={t(lang, "inv.scenario_base_sub")}
                  rent={result.rent_evaluation.predicted_rent_kzt}
                  metrics={inv}
                />
                <ScenarioCard
                  lang={lang}
                  variant="optimistic"
                  title={t(lang, "inv.scenario_optimistic")}
                  subtitle={t(lang, "inv.scenario_optimistic_sub")}
                  rent={result.rent_evaluation.rent_interval.high_kzt}
                  metrics={result.investment_optimistic ?? null}
                />
              </div>
              <div className="callout-amber" style={{ marginBottom: "1.6rem" }}>
                ⚠ {t(lang, "inv.scenarios_warning")}
              </div>
            </>
          )}

          {/* Lead */}
          {narr.lead && (
            <>
              <div className="section-title">{t(lang, "inv.r.lead")}</div>
              <div className="result-block">{narr.lead}</div>
            </>
          )}

          {/* Price paragraph */}
          {narr.price_paragraph && (
            <>
              <div className="section-title">{t(lang, "inv.r.price")}</div>
              <div className="result-block">{narr.price_paragraph}</div>
            </>
          )}

          {/* Investment paragraph */}
          {narr.investment_paragraph && (
            <>
              <div className="section-title">{t(lang, "inv.r.invest")}</div>
              <div className="result-block">{narr.investment_paragraph}</div>
            </>
          )}

          {/* Market context */}
          {narr.market_context_paragraph && (
            <>
              <div className="section-title">{t(lang, "inv.r.market")}</div>
              <div className="result-block">{narr.market_context_paragraph}</div>
            </>
          )}

          {/* Location */}
          {narr.location_paragraph && (
            <>
              <div className="section-title">{t(lang, "inv.r.location")}</div>
              <div className="result-block">{narr.location_paragraph}</div>
            </>
          )}

          {/* Risks */}
          {narr.risks && narr.risks.length > 0 && (
            <>
              <div className="section-title">{t(lang, "inv.r.risks")}</div>
              <div className="surface">
                <ul className="cons-list">
                  {narr.risks.map((r, i) => <li key={i}>{r}</li>)}
                </ul>
              </div>
            </>
          )}

          {/* Verdict text */}
          {narr.verdict_text && (
            <>
              <div className="section-title">{t(lang, "inv.r.verdict")}</div>
              <div className="result-block">{narr.verdict_text}</div>
            </>
          )}

          {/* Listing link */}
          <a href={result.source_url} target="_blank" rel="noopener noreferrer" className="listing-link">
            🔗 {t(lang, "inv.r.link")}
          </a>
        </div>
      )}
    </div>
  );
}
