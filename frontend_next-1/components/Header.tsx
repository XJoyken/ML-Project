"use client";

import Link from "next/link";
import LanguageToggle from "./LanguageToggle";

export default function Header() {
  return (
    <header
      style={{
        background: "var(--bg)",
        borderBottom: "1px solid var(--border)",
        padding: "0.85rem 1.5rem",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        flexShrink: 0,
        position: "sticky",
        top: 0,
        zIndex: 50,
        backdropFilter: "blur(8px)",
        WebkitBackdropFilter: "blur(8px)",
      }}
    >
      <Link
        href="/"
        style={{
          textDecoration: "none",
          display: "flex",
          alignItems: "baseline",
          gap: "0.5rem",
        }}
      >
        <span style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--accent)", letterSpacing: "-0.01em" }}>
          AlmatyNest
        </span>
        <span style={{ fontSize: "0.88rem", fontWeight: 500, color: "var(--text-muted)" }}>
          Intelligence
        </span>
      </Link>

      <LanguageToggle />
    </header>
  );
}
