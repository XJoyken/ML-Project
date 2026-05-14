"use client";

import { useState } from "react";
import { useLang } from "@/context/LanguageContext";
import { t } from "@/lib/i18n";
import { evaluate, ApiUnavailable, ApiBadRequest } from "@/lib/api";
import type { EvaluateResponse } from "@/lib/types";
import BackButton from "@/components/BackButton";
import VerdictBadge from "@/components/VerdictBadge";
import LoadingOverlay from "@/components/LoadingOverlay";
import FreshnessWarning from "@/components/FreshnessWarning";
import Tooltip from "@/components/Tooltip";

function fmtPrice(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)} млн ₸`;
  return `${(n / 1_000).toFixed(0)} тыс ₸`;
}

function fmtDelta(d: number): string {
  const sign = d >= 0 ? "+" : "";
  return `${sign}${d.toFixed(1)}%`;
}

function isKrishaUrl(url: string): boolean {
  return /krisha\.kz\/a\/show\/\d+/.test(url);
}

export default function EvaluatePage() {
  const { lang } = useLang();
  const [url, setUrl] = useState("");
  const [useLlm, setUseLlm] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EvaluateResponse | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);

    if (!url.trim()) { setError(t(lang, "err.url_required")); return; }
    if (!isKrishaUrl(url)) { setError(t(lang, "err.url_invalid")); return; }

    setLoading(true);
    try {
      const data = await evaluate({ url, language: lang, use_llm: useLlm });
      setResult(data);
    } catch (err) {
      if (err instanceof ApiUnavailable) setError(t(lang, "err.unavailable"));
      else if (err instanceof ApiBadRequest) {
        const msg = (err as { message: string }).message;
        setError(msg.includes("not found") || msg.includes("404")
          ? t(lang, "err.not_found")
          : t(lang, "err.bad_request", { detail: msg }));
      } else {
        setError(t(lang, "err.unknown", { detail: String((err as Error).message) }));
      }
    } finally {
      setLoading(false);
    }
  }

  const ev = result?.evaluation;
  const narrative = ev?.narrative;

  return (
    <div className="page-col fade-up">
      <BackButton />
      <h1 style={{ fontSize: "1.7rem", fontWeight: 800, marginBottom: "0.3rem" }}>
        {t(lang, "ev.title")}
      </h1>
      <p style={{ color: "var(--text-muted)", marginBottom: "1.8rem", fontSize: "0.95rem" }}>
        krisha.kz → {lang === "ru" ? "справедливая цена и аналитика района" : "fair price & neighborhood analysis"}
      </p>

      <FreshnessWarning />

      {/* Form */}
      <form onSubmit={handleSubmit} style={{ marginBottom: "1.5rem" }}>
        <div style={{ marginBottom: "1rem" }}>
          <label className="input-label" htmlFor="ev-url">{t(lang, "ev.url_label")}</label>
          <input
            id="ev-url"
            className="input-field"
            type="url"
            placeholder={t(lang, "ev.url_placeholder")}
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            autoComplete="off"
          />
        </div>

        {/* LLM toggle */}
        <div className="toggle-row" style={{ marginBottom: "1.2rem" }}>
          <label className="toggle-switch" htmlFor="ev-llm">
            <input
              id="ev-llm"
              type="checkbox"
              checked={useLlm}
              onChange={(e) => setUseLlm(e.target.checked)}
            />
            <div className="toggle-track" />
          </label>
          <div className="toggle-label-group">
            <span className="toggle-label">{t(lang, "ev.use_llm")}</span>
            <span className="toggle-help">{t(lang, "ev.use_llm_help")}</span>
          </div>
        </div>

        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? "..." : t(lang, "nav.submit")}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div className="callout-danger" style={{ marginBottom: "1rem" }}>{error}</div>
      )}

      {/* Loading */}
      {loading && <LoadingOverlay />}

      {/* Fallback notice */}
      {!loading && narrative?.source === "fallback" && narrative.fallback_reason && (
        <div className="callout-amber" style={{ marginBottom: "1rem" }}>
          {t(lang, "err.fallback_notice")}{narrative.fallback_reason}
        </div>
      )}

      {/* Results */}
      {!loading && ev && narrative && (
        <div className="fade-up">
          {/* Verdict */}
          <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1.4rem" }}>
            <VerdictBadge verdict={narrative.final_verdict ?? ev.verdict} />
          </div>

          {/* Stats */}
          <div className="stat-grid">
            <div className="stat-card">
              <div className="stat-label">{t(lang, "ev.r.predicted")}</div>
              <div className="stat-value">{fmtPrice(ev.predicted_price_kzt)}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">{t(lang, "ev.r.listed")}</div>
              <div className="stat-value">{fmtPrice(ev.listing_price_kzt)}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">{t(lang, "ev.r.delta")}</div>
              <div
                className="stat-value"
                style={{ color: ev.delta_percent > 0 ? "var(--success)" : "var(--danger)" }}
              >
                {fmtDelta(ev.delta_percent)}
              </div>
            </div>
            {ev.price_interval && (
              <div className="stat-card">
                <div className="stat-label" style={{ display: "flex", alignItems: "center" }}>
                  {t(lang, "ev.r.interval")}
                  <Tooltip text={t(lang, "ev.r.interval_tooltip")} />
                </div>
                <div className="stat-value" style={{ fontSize: "0.95rem" }}>
                  {fmtPrice(ev.price_interval.low_kzt)} – {fmtPrice(ev.price_interval.high_kzt)}
                </div>
              </div>
            )}
          </div>

          {/* Lead */}
          {narrative.lead && (
            <div className="result-block" style={{ marginBottom: "1rem" }}>
              {narrative.lead}
            </div>
          )}

          {/* Pros / Cons */}
          {((narrative.pros?.length ?? 0) > 0 || (narrative.cons?.length ?? 0) > 0) && (
            <>
              <div className="section-title">
                {t(lang, "ev.r.pros")} / {t(lang, "ev.r.cons")}
              </div>
              <div className="two-col">
                <div className="surface">
                  <div className="section-title" style={{ marginTop: 0 }}>
                    ✓ {t(lang, "ev.r.pros")}
                  </div>
                  <ul className="pros-list">
                    {narrative.pros?.map((p, i) => <li key={i}>{p}</li>)}
                  </ul>
                </div>
                <div className="surface">
                  <div className="section-title" style={{ marginTop: 0 }}>
                    ✗ {t(lang, "ev.r.cons")}
                  </div>
                  <ul className="cons-list">
                    {narrative.cons?.map((c, i) => <li key={i}>{c}</li>)}
                  </ul>
                </div>
              </div>
            </>
          )}

          {/* Persona notes */}
          {narrative.persona_notes && narrative.persona_notes.length > 0 && (
            <>
              <div className="section-title">{t(lang, "ev.r.persona")}</div>
              <div className="surface">
                <ul className="persona-list">
                  {narrative.persona_notes.map((p, i) => <li key={i}>{p}</li>)}
                </ul>
              </div>
            </>
          )}

          {/* Final verdict — show the localized sentence, not the key */}
          {narrative.final_verdict_text && (
            <>
              <div className="section-title">{t(lang, "ev.r.verdict")}</div>
              <div className="result-block">{narrative.final_verdict_text}</div>
            </>
          )}

          {/* Listing link */}
          <a href={result.source_url} target="_blank" rel="noopener noreferrer" className="listing-link">
            🔗 {t(lang, "ev.r.link")}
          </a>
        </div>
      )}
    </div>
  );
}
