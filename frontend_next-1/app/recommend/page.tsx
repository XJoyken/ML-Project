"use client";

import { useState } from "react";
import { useLang } from "@/context/LanguageContext";
import { t } from "@/lib/i18n";
import { recommend, ApiUnavailable, ApiBadRequest } from "@/lib/api";
import type { RecommendResponse, RecommendItem } from "@/lib/types";
import BackButton from "@/components/BackButton";
import LoadingOverlay from "@/components/LoadingOverlay";
import FreshnessWarning from "@/components/FreshnessWarning";
import { fmtRooms, fmtArea, fmtFloor } from "@/lib/format";

function fmtPrice(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)} млн ₸`;
  return `${(n / 1_000).toFixed(0)} тыс ₸`;
}

function fmtScore(s: number): string {
  return `${(s * 100).toFixed(0)}%`;
}

function ListingCard({ item, lang }: { item: RecommendItem; lang: "ru" | "en" }) {
  const url = item.url ?? `https://krisha.kz/a/show/${item.listing_id}`;
  const price = item.price_kzt;
  const area = item.area_m2;
  return (
    <div className="listing-card">
      <div className="listing-card-header">
        <span style={{ fontWeight: 700, fontSize: "1rem" }}>
          {price !== null && price !== undefined ? fmtPrice(price) : "—"}
        </span>
        <span className="listing-card-score">{t(lang, "rc.score")} {fmtScore(item.match_score)}</span>
      </div>

      {/* Key stats row: rooms → area → floor → district */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "1rem",
          fontSize: "0.87rem",
          color: "var(--text-muted)",
          marginBottom: "0.7rem",
        }}
      >
        {item.rooms != null && <span>🛏 {fmtRooms(item.rooms, lang)}</span>}
        {area != null && <span>📐 {fmtArea(area)}</span>}
        {fmtFloor(item.floor_current, item.floors_total, lang) && (
          <span>🏢 {fmtFloor(item.floor_current, item.floors_total, lang)}</span>
        )}
        {item.district && <span>📍 {item.district}</span>}
      </div>

      {/* AI explanation */}
      {item.explanation && (
        <p style={{ fontSize: "0.9rem", lineHeight: 1.55, color: "var(--text)", margin: "0 0 0.8rem 0" }}>
          {item.explanation}
        </p>
      )}

      <a href={url} target="_blank" rel="noopener noreferrer" className="listing-link" style={{ marginTop: 0 }}>
        {t(lang, "rc.open")}
      </a>
    </div>
  );
}

export default function RecommendPage() {
  const { lang } = useLang();
  const [prompt, setPrompt] = useState("");
  const [limit, setLimit] = useState(7);
  const [mmrLambda, setMmrLambda] = useState(0.7);
  const [airPriority, setAirPriority] = useState(false);
  const [useLlm, setUseLlm] = useState(true);
  const [showParams, setShowParams] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RecommendResponse | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);

    if (prompt.trim().length < 3) { setError(t(lang, "err.prompt_short")); return; }

    setLoading(true);
    try {
      const data = await recommend({
        prompt,
        limit,
        language: lang,
        use_llm: useLlm,
        mmr_lambda: mmrLambda,
        prioritize_air_quality: airPriority,
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

  return (
    <div className="page-col fade-up">
      <BackButton />
      <h1 style={{ fontSize: "1.7rem", fontWeight: 800, marginBottom: "0.3rem" }}>
        {t(lang, "rc.title")}
      </h1>
      <p style={{ color: "var(--text-muted)", marginBottom: "1.8rem", fontSize: "0.95rem" }}>
        {lang === "ru"
          ? "Опишите пожелания — модель подберёт подходящие квартиры"
          : "Describe your needs — the model will match apartments for you"}
      </p>

      <FreshnessWarning />

      <form onSubmit={handleSubmit} style={{ marginBottom: "1.5rem" }}>
        {/* Prompt */}
        <div style={{ marginBottom: "1rem" }}>
          <label className="input-label" htmlFor="rc-prompt">{t(lang, "rc.prompt_label")}</label>
          <textarea
            id="rc-prompt"
            className="input-field"
            placeholder={t(lang, "rc.prompt_placeholder")}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
          />
        </div>

        {/* Advanced params expander */}
        <div className="expander" style={{ marginBottom: "1.2rem" }}>
          <button
            type="button"
            className="expander-header"
            onClick={() => setShowParams(!showParams)}
          >
            <span>⚙ {t(lang, "rc.params")}</span>
            <span style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
              {showParams ? "▲" : "▼"}
            </span>
          </button>
          {showParams && (
            <div className="expander-body">
              {/* Limit */}
              <div style={{ marginBottom: "1rem" }}>
                <label className="input-label" htmlFor="rc-limit">
                  {t(lang, "rc.limit")} — {limit}
                </label>
                <input
                  id="rc-limit"
                  type="range"
                  min={3} max={15} step={1}
                  value={limit}
                  onChange={(e) => setLimit(+e.target.value)}
                  style={{ width: "100%", accentColor: "var(--accent)" }}
                />
                <div className="toggle-help" style={{ marginTop: "0.2rem" }}>{t(lang, "rc.limit_help")}</div>
              </div>

              {/* MMR lambda */}
              <div style={{ marginBottom: "1rem" }}>
                <label className="input-label" htmlFor="rc-mmr">
                  {t(lang, "rc.mmr")} — {mmrLambda.toFixed(2)}
                </label>
                <input
                  id="rc-mmr"
                  type="range"
                  min={0} max={1} step={0.05}
                  value={mmrLambda}
                  onChange={(e) => setMmrLambda(+e.target.value)}
                  style={{ width: "100%", accentColor: "var(--accent)" }}
                />
                <div className="toggle-help" style={{ marginTop: "0.2rem" }}>{t(lang, "rc.mmr_help")}</div>
              </div>

              {/* Air priority */}
              <div className="toggle-row" style={{ marginBottom: "0.8rem" }}>
                <label className="toggle-switch" htmlFor="rc-air">
                  <input id="rc-air" type="checkbox" checked={airPriority} onChange={(e) => setAirPriority(e.target.checked)} />
                  <div className="toggle-track" />
                </label>
                <div className="toggle-label-group">
                  <span className="toggle-label">{t(lang, "rc.air")}</span>
                  <span className="toggle-help">{t(lang, "rc.air_help")}</span>
                </div>
              </div>

              {/* Use LLM */}
              <div className="toggle-row">
                <label className="toggle-switch" htmlFor="rc-llm">
                  <input id="rc-llm" type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
                  <div className="toggle-track" />
                </label>
                <div className="toggle-label-group">
                  <span className="toggle-label">{t(lang, "rc.use_llm")}</span>
                  <span className="toggle-help">{t(lang, "rc.use_llm_help")}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? "..." : t(lang, "rc.submit")}
        </button>
      </form>

      {/* Error */}
      {error && <div className="callout-danger" style={{ marginBottom: "1rem" }}>{error}</div>}

      {/* Loading */}
      {loading && <LoadingOverlay />}

      {/* Results */}
      {!loading && result && (
        <div className="fade-up">
          {/* Unmapped preferences */}
          {result.fallback_reason && (
            <div className="callout-amber" style={{ marginBottom: "1rem" }}>
              {t(lang, "err.fallback_notice")}{result.fallback_reason}
            </div>
          )}

          {/* Unmapped preferences from features */}
          {Array.isArray((result.features as { unmapped_preferences?: string[] })?.unmapped_preferences) &&
           ((result.features as { unmapped_preferences?: string[] }).unmapped_preferences?.length ?? 0) > 0 && (
            <div className="callout-amber" style={{ marginBottom: "1rem" }}>
              {t(lang, "rc.unmapped")}
              {(result.features as { unmapped_preferences?: string[] }).unmapped_preferences?.join(", ")}
            </div>
          )}

          {/* Summary */}
          <div className="section-title">{t(lang, "rc.summary_title")}</div>
          <div className="result-block">{result.summary}</div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "1.2rem" }}>
            <span style={{ color: "var(--text-muted)", fontSize: "0.87rem" }}>
              {t(lang, "rc.matched")}:
            </span>
            <span style={{ color: "var(--accent)", fontWeight: 700, fontSize: "0.9rem" }}>
              {result.matched_candidates}
            </span>
          </div>

          {/* Listing cards */}
          {result.items.map((item, i) => (
            <ListingCard key={String(item.listing_id) + i} item={item} lang={lang} />
          ))}
        </div>
      )}
    </div>
  );
}
