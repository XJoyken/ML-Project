"use client";

import { useLang } from "@/context/LanguageContext";
import { t } from "@/lib/i18n";

const VERDICT_KEYS: Record<string, string> = {
  excellent:    "v.excellent",
  good:         "v.good",
  average:      "v.average",
  questionable: "v.questionable",
  poor:         "v.poor",
};

export default function VerdictBadge({ verdict }: { verdict: string }) {
  const { lang } = useLang();
  const key = VERDICT_KEYS[verdict?.toLowerCase()] ?? "v.average";
  const label = t(lang, key);
  return (
    <span className={`badge badge-${verdict?.toLowerCase() ?? "average"}`}>
      {label}
    </span>
  );
}
