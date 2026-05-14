"use client";

import { useRouter } from "next/navigation";
import { useLang } from "@/context/LanguageContext";
import { t } from "@/lib/i18n";

export default function BackButton() {
  const router = useRouter();
  const { lang } = useLang();
  return (
    <button className="btn-secondary" onClick={() => router.back()} style={{ marginBottom: "1.4rem" }}>
      {t(lang, "nav.back")}
    </button>
  );
}
