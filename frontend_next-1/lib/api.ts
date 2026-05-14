import type {
  EvaluateResponse,
  RecommendResponse,
  InvestmentResponse,
  InvestmentParams,
} from "./types";

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
const TIMEOUT = 90_000;

class ApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = "ApiError";
  }
}
export class ApiUnavailable extends ApiError { constructor(msg: string) { super(msg); this.name = "ApiUnavailable"; } }
export class ApiBadRequest  extends ApiError { constructor(msg: string, status: number) { super(msg, status); this.name = "ApiBadRequest"; } }
export class ApiServerError extends ApiError { constructor(msg: string, status: number) { super(msg, status); this.name = "ApiServerError"; } }

async function post<T>(path: string, body: unknown): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT);
  let res: Response;
  try {
    res = await fetch(`${BACKEND}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err: unknown) {
    clearTimeout(timer);
    const msg = err instanceof Error ? err.message : String(err);
    throw new ApiUnavailable(msg);
  }
  clearTimeout(timer);

  if (res.status >= 500) {
    const detail = await extractDetail(res);
    throw new ApiServerError(detail, res.status);
  }
  if (res.status >= 400) {
    const detail = await extractDetail(res);
    throw new ApiBadRequest(detail, res.status);
  }
  return res.json() as Promise<T>;
}

async function extractDetail(res: Response): Promise<string> {
  try {
    const payload = await res.json();
    if (typeof payload === "object" && payload !== null) {
      const d = (payload as Record<string, unknown>).detail;
      if (typeof d === "string") return d;
      if (Array.isArray(d) && d.length > 0) {
        const first = d[0] as Record<string, unknown>;
        return (typeof first.msg === "string" ? first.msg : null) ?? String(d[0]);
      }
    }
  } catch { /* ignore */ }
  return `HTTP ${res.status}`;
}

export function evaluate(args: {
  url: string;
  language: "ru" | "en";
  use_llm: boolean | null;
}): Promise<EvaluateResponse> {
  return post("/apartments/evaluate", args);
}

export function recommend(args: {
  prompt: string;
  limit: number;
  language: "ru" | "en";
  use_llm: boolean | null;
  mmr_lambda: number;
  prioritize_air_quality: boolean;
}): Promise<RecommendResponse> {
  return post("/recommendations", args);
}

export function investment(args: {
  url: string;
  language: "ru" | "en";
  use_llm: boolean | null;
  investment_params: InvestmentParams | null;
}): Promise<InvestmentResponse> {
  return post("/apartments/investment", args);
}
