export interface NarrativeReport {
  language?: string;
  lead: string;
  pros: string[];
  cons: string[];
  persona_notes: string[];
  final_verdict: string;          // key: "excellent" | "good" | "questionable" | "poor"
  final_verdict_text: string;     // sentence in the requested language
  source: "gemini" | "fallback";
  fallback_reason_code?: string | null;
  fallback_reason?: string | null;
}

export interface PriceInterval {
  coverage: number;
  low_kzt: number;
  high_kzt: number;
}

export interface SaleEvaluation {
  model: string;
  predicted_price_kzt: number;
  listing_price_kzt: number;
  delta_percent: number;
  price_interval: PriceInterval;
  verdict: string;        // legacy verdict from get_verdict()
  final_verdict: string;  // narrative's verdict key
  narrative: NarrativeReport;
}

export interface EvaluateResponse {
  source_url: string;
  parsed_listing: Record<string, unknown>;
  evaluation: SaleEvaluation;
}

// ── Recommend ──────────────────────────────────────────────────────────

export interface RecommendItem {
  listing_id: string | number;
  url: string;
  match_score: number;
  base_score?: number;
  air_bonus?: number;
  price_kzt: number | null;
  area_m2: number | null;
  rooms: number | null;
  floor_current?: number | null;
  floors_total?: number | null;
  year_built?: number | null;
  condition?: string | null;
  house_type?: string | null;
  district: string | null;
  microdistrict?: string | null;
  explanation: string;
  description?: string | null;
  [key: string]: unknown;
}

export interface ExtractedFeatures {
  [key: string]: unknown;
}

export interface RecommendResponse {
  model: string;
  features: ExtractedFeatures;
  items: RecommendItem[];
  plan: Record<string, unknown>;
  summary: string;
  source: "gemini" | "fallback";
  fallback_reason_code?: string | null;
  fallback_reason?: string | null;
  matched_candidates: number;
}

// ── Investment ─────────────────────────────────────────────────────────

export interface InvestmentMetrics {
  sale_price_kzt: number;
  monthly_rent_kzt: number;
  total_investment_kzt: number;
  annual_gross_rent_kzt: number;
  annual_recurring_costs_kzt: number;
  annual_net_cashflow_kzt: number;
  gross_yield_pct: number;
  net_yield_pct: number;
  payback_years_nominal: number | null;
  payback_years_inflation_adjusted: number | null;
  npv_kzt: number;
  irr_pct: number | null;
  market_yield_pct: number | null;
  yield_vs_market_pct_points: number | null;
  verdict: string;
  params?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface InvestmentNarrative {
  lead: string;
  price_paragraph: string;
  investment_paragraph: string;
  market_context_paragraph: string;
  location_paragraph: string;
  risks: string[];
  verdict: string;
  verdict_text: string;
  source: "gemini" | "fallback";
  fallback_reason?: string | null;
}

export interface RentInterval {
  coverage: number;
  low_kzt: number;
  high_kzt: number;
}

export interface RentEvaluation {
  predicted_rent_kzt: number;
  rent_interval?: RentInterval;
  [key: string]: unknown;
}

export interface InvestmentResponse {
  source_url: string;
  parsed_listing: Record<string, unknown>;
  sale_evaluation: SaleEvaluation;
  rent_evaluation: RentEvaluation;
  investment: InvestmentMetrics;
  // Sensitivity scenarios — same params, rent at the 90% conformal-interval bounds.
  investment_optimistic?: InvestmentMetrics | null;
  investment_conservative?: InvestmentMetrics | null;
  market_summary: Record<string, unknown>;
  narrative: InvestmentNarrative;
}

// ── Investment params ──────────────────────────────────────────────────

export interface InvestmentParams {
  vacancy_rate?: number | null;
  repair_cost_pct?: number | null;
  agent_commission_months?: number | null;
  tenant_turnover_years?: number | null;
  maintenance_pct?: number | null;
  property_tax_pct?: number | null;
  inflation_rate_pct?: number | null;
  risk_premium_pct?: number | null;
  horizon_years?: number | null;
}
