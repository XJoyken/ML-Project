import type { Lang } from "@/context/LanguageContext";

/** Russian pluralization picker. forms = [one, few, many]. */
export function pluralRu(n: number, forms: [string, string, string]): string {
  const abs = Math.abs(Math.round(n));
  const n10 = abs % 10;
  const n100 = abs % 100;
  if (n10 === 1 && n100 !== 11) return forms[0];
  if (n10 >= 2 && n10 <= 4 && (n100 < 12 || n100 > 14)) return forms[1];
  return forms[2];
}

/** "2 комнаты" / "2 rooms" — language-aware. */
export function fmtRooms(n: number | null | undefined, lang: Lang): string {
  if (n == null) return "—";
  const v = Math.round(n);
  if (lang === "ru") {
    return `${v} ${pluralRu(v, ["комната", "комнаты", "комнат"])}`;
  }
  return `${v} ${v === 1 ? "room" : "rooms"}`;
}

/** "42.3 м²" / "42.3 m²" */
export function fmtArea(n: number | null | undefined): string {
  if (n == null) return "—";
  return `${n.toFixed(1)} м²`;
}

/** "4/9" floor or null. */
export function fmtFloor(
  current: number | null | undefined,
  total: number | null | undefined,
  lang: Lang,
): string | null {
  if (current == null) return null;
  const c = Math.round(current);
  const t = total == null ? null : Math.round(total);
  const label = lang === "ru" ? "этаж" : "floor";
  return t != null ? `${c}/${t} ${label}` : `${c} ${label}`;
}
