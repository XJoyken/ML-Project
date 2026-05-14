"use client";

import { useState, useRef, useLayoutEffect, useEffect } from "react";

interface Pos { top: number; left: number; arrowLeft: number; placement: "top" | "bottom"; }

export default function Tooltip({ text }: { text: string }) {
  const triggerRef = useRef<HTMLSpanElement>(null);
  const tipRef = useRef<HTMLSpanElement>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<Pos>({ top: 0, left: 0, arrowLeft: 50, placement: "top" });

  useLayoutEffect(() => {
    if (!open || !triggerRef.current || !tipRef.current) return;
    const trigger = triggerRef.current.getBoundingClientRect();
    const tip = tipRef.current.getBoundingClientRect();
    const padding = 8;
    const gap = 10;

    const triggerCenterX = trigger.left + trigger.width / 2;

    // Horizontal: centered on trigger, clamped to viewport
    let left = triggerCenterX - tip.width / 2;
    if (left < padding) left = padding;
    if (left + tip.width > window.innerWidth - padding) {
      left = window.innerWidth - tip.width - padding;
    }

    // Vertical: prefer above. Flip to below if too close to top edge.
    let placement: "top" | "bottom" = "top";
    let top = trigger.top - tip.height - gap;
    if (top < padding) {
      placement = "bottom";
      top = trigger.bottom + gap;
    }

    // Arrow position: relative to tooltip width
    const arrowLeftPx = triggerCenterX - left;
    const arrowLeft = Math.max(10, Math.min(tip.width - 10, arrowLeftPx));

    setPos({ top, left, arrowLeft, placement });
  }, [open]);

  // Close on scroll (otherwise the position becomes stale)
  useEffect(() => {
    if (!open) return;
    const close = () => setOpen(false);
    window.addEventListener("scroll", close, true);
    return () => window.removeEventListener("scroll", close, true);
  }, [open]);

  return (
    <>
      <span
        ref={triggerRef}
        className="tooltip-trigger"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        tabIndex={0}
        role="tooltip"
        aria-label={text}
      >
        ?
      </span>
      {open && (
        <span
          ref={tipRef}
          className={`tooltip-body tooltip-body--${pos.placement}`}
          style={{ top: pos.top, left: pos.left }}
          role="presentation"
        >
          {text}
          <span className="tooltip-arrow" style={{ left: pos.arrowLeft }} />
        </span>
      )}
    </>
  );
}
