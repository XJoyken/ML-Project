"use client";

import { useLang } from "@/context/LanguageContext";
import { t } from "@/lib/i18n";

export default function Footer() {
  const { lang } = useLang();
  return (
    <footer
      style={{
        borderTop: "1px solid var(--border)",
        padding: "1rem 1.5rem",
        textAlign: "center",
        color: "var(--text-muted)",
        fontSize: "0.82rem",
        lineHeight: 1.5,
        flexShrink: 0,
        background: "var(--bg)",
      }}
    >
      {t(lang, "footer.disclaimer")}
    </footer>
  );
}
