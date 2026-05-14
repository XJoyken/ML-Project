"use client";

import { useRef, useLayoutEffect, useState } from "react";
import { useLang } from "@/context/LanguageContext";

export default function LanguageToggle() {
  const { lang, setLang } = useLang();
  const ruRef = useRef<HTMLButtonElement>(null);
  const enRef = useRef<HTMLButtonElement>(null);
  const [slider, setSlider] = useState({ left: 3, width: 0 });

  useLayoutEffect(() => {
    const btn = lang === "ru" ? ruRef.current : enRef.current;
    if (btn) {
      setSlider({ left: btn.offsetLeft, width: btn.offsetWidth });
    }
  }, [lang]);

  return (
    <div className="lang-toggle" aria-label="Language switcher">
      <div
        className="lang-toggle-slider"
        style={{ left: slider.left, width: slider.width || undefined }}
      />
      <button
        ref={ruRef}
        className={`lang-toggle-btn${lang === "ru" ? " active" : ""}`}
        onClick={() => setLang("ru")}
        aria-pressed={lang === "ru"}
      >
        RU
      </button>
      <button
        ref={enRef}
        className={`lang-toggle-btn${lang === "en" ? " active" : ""}`}
        onClick={() => setLang("en")}
        aria-pressed={lang === "en"}
      >
        EN
      </button>
    </div>
  );
}
