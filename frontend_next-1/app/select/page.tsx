"use client";

import Link from "next/link";
import { useLang } from "@/context/LanguageContext";
import { t } from "@/lib/i18n";
import BackButton from "@/components/BackButton";
import FreshnessWarning from "@/components/FreshnessWarning";

const MODELS = [
  { icon: "🏠", titleKey: "m1.title", descKey: "m1.desc", href: "/evaluate" },
  { icon: "🔍", titleKey: "m2.title", descKey: "m2.desc", href: "/recommend" },
  { icon: "💼", titleKey: "m3.title", descKey: "m3.desc", href: "/investment" },
];

export default function SelectPage() {
  const { lang } = useLang();
  return (
    <div className="page-col fade-up">
      <BackButton />

      <h1 style={{ fontSize: "1.85rem", fontWeight: 800, marginBottom: "0.4rem" }}>
        {t(lang, "select.title")}
      </h1>
      <p style={{ color: "var(--text-muted)", fontSize: "0.96rem", marginBottom: "1.8rem" }}>
        {t(lang, "select.subtitle")}
      </p>

      <FreshnessWarning />

      <div className="model-cards-grid" style={{ marginBottom: "3rem" }}>
        {MODELS.map((m) => (
          <Link key={m.href} href={m.href} className="model-card">
            <div className="model-card-icon">{m.icon}</div>
            <div className="model-card-title">{t(lang, m.titleKey)}</div>
            <div className="model-card-desc">{t(lang, m.descKey)}</div>
            <div className="model-card-arrow">
              {t(lang, "card.open")}
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
