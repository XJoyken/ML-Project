"use client";

import Link from "next/link";
import { useLang } from "@/context/LanguageContext";
import { t } from "@/lib/i18n";

export default function LandingPage() {
  const { lang } = useLang();

  return (
    <div className="page-col fade-up" style={{ minHeight: "calc(100vh - 140px)" }}>
      {/* Hero */}
      <div style={{ paddingTop: "3.5rem", paddingBottom: "2.5rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "1rem" }}>
          <span
            style={{
              fontSize: "0.78rem",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.12em",
              color: "var(--accent)",
              background: "rgba(232,215,124,0.1)",
              padding: "0.25rem 0.75rem",
              borderRadius: "999px",
            }}
          >
            {t(lang, "app.tagline")}
          </span>
        </div>

        <h1 className="hero-title">{t(lang, "app.name")}</h1>
        <p className="hero-sub" style={{ marginTop: "0.5rem" }}>{t(lang, "landing.subtitle")}</p>
      </div>

      {/* Feature highlights */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: "1rem",
          marginBottom: "2.5rem",
        }}
      >
        {[
          { icon: "🏠", titleKey: "landing.f1.title", descKey: "landing.f1.desc" },
          { icon: "🔍", titleKey: "landing.f2.title", descKey: "landing.f2.desc" },
          { icon: "💼", titleKey: "landing.f3.title", descKey: "landing.f3.desc" },
        ].map((f, i) => (
          <div
            key={i}
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: "14px",
              padding: "1.2rem 1.4rem",
            }}
          >
            <div style={{ fontSize: "1.7rem", marginBottom: "0.5rem" }}>{f.icon}</div>
            <div style={{ fontWeight: 700, fontSize: "1rem", marginBottom: "0.35rem" }}>
              {t(lang, f.titleKey)}
            </div>
            <div style={{ color: "var(--text-muted)", fontSize: "0.87rem", lineHeight: 1.5 }}>
              {t(lang, f.descKey)}
            </div>
          </div>
        ))}
      </div>

      {/* Tech stack accent bar */}
      <div
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "14px",
          padding: "1.4rem 1.8rem",
          marginBottom: "2.5rem",
        }}
      >
        <div style={{ fontWeight: 700, fontSize: "1rem", marginBottom: "0.35rem" }}>
          {t(lang, "landing.tech.title")}
        </div>
        <div style={{ color: "var(--text-muted)", fontSize: "0.88rem", lineHeight: 1.55 }}>
          {t(lang, "landing.tech.desc")}
        </div>
      </div>

      {/* CTA */}
      <div style={{ display: "flex", justifyContent: "center", padding: "1rem 0 2.5rem" }}>
        <Link href="/select" className="btn-primary" style={{ fontSize: "1.05rem", padding: "0.85rem 2.2rem" }}>
          {t(lang, "nav.start")}
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </Link>
      </div>
    </div>
  );
}
