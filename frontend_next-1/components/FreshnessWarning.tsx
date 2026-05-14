"use client";

import { useLang } from "@/context/LanguageContext";
import { t } from "@/lib/i18n";

export default function FreshnessWarning() {
  const { lang } = useLang();
  return (
    <div className="callout-danger" style={{ marginBottom: "1.5rem" }}>
      {t(lang, "warning.freshness")}
    </div>
  );
}
