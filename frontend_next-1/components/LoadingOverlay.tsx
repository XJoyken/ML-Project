"use client";

import { useState, useEffect } from "react";
import { useLang } from "@/context/LanguageContext";
import { t, LOADER_KEYS } from "@/lib/i18n";

export default function LoadingOverlay() {
  const { lang } = useLang();
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setIdx((i) => (i + 1) % LOADER_KEYS.length);
    }, 3000);
    return () => clearInterval(id);
  }, []);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "1.2rem",
        padding: "3rem 1rem",
      }}
    >
      <div className="spinner" />
      <p
        key={idx}
        className="loading-msg"
        style={{ color: "var(--text-muted)", fontSize: "0.95rem", margin: 0 }}
      >
        {t(lang, LOADER_KEYS[idx])}
      </p>
    </div>
  );
}
