import type { Lang } from "@/context/LanguageContext";

const T: Record<Lang, Record<string, string>> = {
  ru: {
    "app.name": "AlmatyNest Intelligence",
    "app.tagline": "Алматы · ИИ-анализ недвижимости",
    "nav.back": "← Назад",
    "nav.start": "Начать",
    "nav.submit": "Проверить",
    "footer.disclaimer":
      "Все оценки даются как ориентир, а не финансовый совет. Модели могут ошибаться — всегда проверяйте информацию самостоятельно.",
    "warning.freshness":
      "⚠ Модели обучены на данных до марта 2026 г. Прогнозы могут не учитывать свежие изменения рынка.",

    // Landing
    "landing.subtitle":
      "Помогаем оценить квартиру, найти подходящую и понять, стоит ли вкладываться в аренду — три модели на одном сайте.",
    "landing.cta": "Начать",
    "landing.f1.title": "Честная цена",
    "landing.f1.desc": "Проверяем, не переплачиваете ли вы за квартиру по сравнению с рынком.",
    "landing.f2.title": "Подбор под запрос",
    "landing.f2.desc": "Описываете желаемое — модель находит подходящие квартиры из тысяч объявлений.",
    "landing.f3.title": "Доходность аренды",
    "landing.f3.desc": "Считаем окупаемость, NPV, IRR — стоит ли брать квартиру под сдачу.",
    "landing.tech.title": "Что под капотом",
    "landing.tech.desc": "LightGBM, conformal-интервалы, Gemini AI · MAPE 7.25% (продажа), 12.3% (аренда) · 89+ тестов",
    "select.title": "Выберите модель",
    "select.subtitle": "Каждая решает свою задачу — нажмите на блок, чтобы открыть.",
    "m1.title": "Оценка квартиры",
    "m1.desc":
      "Закиньте ссылку на krisha.kz — модель скажет, переплачиваете вы или нет, и сравнит район с другими.",
    "m2.title": "Рекомендации",
    "m2.desc":
      "Опишите квартиру вашей мечты обычным языком — получите 5–10 объявлений, отобранных под ваши пожелания.",
    "m3.title": "Для инвесторов",
    "m3.desc":
      "Узнайте доходность, срок окупаемости и NPV объявления, если вы планируете покупку под сдачу.",
    "card.open": "Открыть →",

    // Evaluate
    "ev.title": "Оценка квартиры",
    "ev.url_label": "Ссылка на krisha.kz",
    "ev.url_placeholder": "https://krisha.kz/a/show/1009196093",
    "ev.use_llm": "Ответ от Gemini",
    "ev.use_llm_help":
      "Если включено — текст оценки сгенерирует ИИ-модель Gemini. При перегрузке автоматически переключается на шаблон.",
    "ev.r.predicted": "Оценка модели",
    "ev.r.listed": "Цена в объявлении",
    "ev.r.delta": "Отклонение",
    "ev.r.interval": "Рыночный диапазон цены",
    "ev.r.interval_tooltip":
      "Диапазон, в котором цена считается справедливой по рынку. Если объявление в этих границах — переплаты нет.",
    "ev.r.pros": "Плюсы",
    "ev.r.cons": "Минусы",
    "ev.r.persona": "Для разных людей",
    "ev.r.verdict": "Финальный вердикт",
    "ev.r.link": "Рассмотренная квартира",

    // Recommend
    "rc.title": "Рекомендации",
    "rc.prompt_label": "Опишите квартиру вашей мечты",
    "rc.prompt_placeholder":
      "Например: двухкомнатную в ЖК до 40 млн ближе к центру, не на первом этаже, рядом школа и парк.",
    "rc.params": "Дополнительные параметры",
    "rc.limit": "Количество вариантов",
    "rc.limit_help": "Сколько квартир вернуть в подборке.",
    "rc.mmr": "Разнообразие ↔ Релевантность",
    "rc.mmr_help":
      "Ближе к 1.0 — точнее по запросу. Ближе к 0.0 — больше разнообразия по районам и ценам.",
    "rc.air": "Приоритизировать чистый воздух",
    "rc.air_help":
      "При равных условиях предпочитать квартиры с меньшим PM2.5.",
    "rc.use_llm": "Ответ от Gemini",
    "rc.use_llm_help":
      "Если включено — Gemini напишет краткое объяснение по каждой квартире.",
    "rc.submit": "Найти квартиры",
    "rc.unmapped": "Модель не учла следующие пожелания (сформулируйте прямее): ",
    "rc.summary_title": "Обзор подборки",
    "rc.matched": "Квартир прошло фильтр",
    "rc.score": "Совпадение",
    "rc.price": "Цена",
    "rc.area": "Площадь",
    "rc.rooms": "Комнат",
    "rc.district": "Район",
    "rc.open": "Открыть на krisha.kz →",

    // Investment
    "inv.title": "Инвестиционный анализ",
    "inv.url_label": "Ссылка на krisha.kz",
    "inv.url_placeholder": "https://krisha.kz/a/show/1009196093",
    "inv.params": "Параметры расчёта",
    "inv.vacancy": "Вакантность",
    "inv.vacancy_help":
      "Доля года, когда квартира стоит пустой между арендаторами. Например 0.08 значит примерно 1 месяц простоя в году.",
    "inv.repair": "Ремонт перед сдачей",
    "inv.repair_help":
      "Единовременные расходы на косметику и мебель перед первой сдачей в аренду. Указывается как доля от стоимости квартиры.",
    "inv.agent": "Комиссия риелтора",
    "inv.agent_help":
      "Сколько месячных арендных плат вы платите риелтору за поиск нового арендатора. На рынке Алматы это обычно 0.5–1 месяц.",
    "inv.turnover": "Срок аренды одним жильцом",
    "inv.turnover_help":
      "Как долго в среднем живёт один арендатор перед тем как съехать. Влияет на то, как часто вы платите риелтору.",
    "inv.maintenance": "Обслуживание",
    "inv.maintenance_help":
      "Текущие расходы на квартиру: мелкий ремонт, замена техники, коммунальные платежи при простое. Доля от годовой арендной платы.",
    "inv.tax": "Налог на имущество",
    "inv.tax_help":
      "Годовой налог на квартиру, как доля от её стоимости. В Казахстане для физлиц обычно 0.1–1% в зависимости от стоимости.",
    "inv.inflation": "Инфляция",
    "inv.inflation_help":
      "Ожидаемая годовая инфляция тенге в процентах. Используется для расчёта реальной окупаемости и ставки дисконтирования.",
    "inv.risk": "Риск-премия",
    "inv.risk_help":
      "Дополнительная доходность в процентных пунктах сверх инфляции, которую вы хотите получить за риск вложения в неликвидную недвижимость. Ставка дисконтирования = инфляция + риск-премия.",
    "inv.horizon": "Горизонт инвестиций",
    "inv.horizon_help": "На сколько лет вперёд считается NPV (чистая приведённая стоимость).",
    "inv.use_llm": "Ответ от Gemini",
    "inv.use_llm_help":
      "Если включено — Gemini напишет аналитический инвестиционный отчёт.",
    "inv.submit": "Рассчитать",
    "inv.r.lead": "Главное",
    "inv.r.price": "Цена и оценка модели",
    "inv.r.invest": "Инвестиционные показатели",
    "inv.r.market": "Контекст рынка",
    "inv.r.location": "Локация для аренды",
    "inv.r.risks": "Риски",
    "inv.r.verdict": "Финальный вердикт",
    "inv.r.link": "Рассмотренная квартира",
    "inv.r.thresholds_title": "Что значит этот вердикт?",
    "inv.r.thresholds_intro":
      "Вердикт строится по окупаемости с учётом инфляции (реальная окупаемость, NPV-метод):",
    "inv.r.threshold_excellent": "Отлично — окупаемость < 8 лет",
    "inv.r.threshold_good": "Хорошо — 8–12 лет",
    "inv.r.threshold_average": "Средне — 12–16 лет",
    "inv.r.threshold_poor": "Плохо — > 16 лет (или валовая доходность < 70% рынка)",
    "inv.r.threshold_current": "Сейчас реальная окупаемость:",
    "inv.r.thresholds_context":
      "В Казахстане инфляция 12.3% и риск-премия 3 пп дают ставку дисконтирования ≈15%. При такой ставке реальная окупаемость почти всегда выходит длинной, поэтому большинство квартир попадает в «средне» или «плохо». Чтобы получить «хорошо», нужна валовая доходность от 10–12%, либо снижение риск-премии (если вы готовы к большему риску), либо более дешёвая входная цена.",
    "inv.scenarios_title": "Диапазон сценариев аренды",
    "inv.scenarios_lead":
      "Как изменится вердикт, если реальная аренда окажется ниже или выше прогноза модели. Эти границы — 90% доверительный интервал по обучающей выборке.",
    "inv.scenario_conservative": "Консервативный",
    "inv.scenario_base": "Базовый",
    "inv.scenario_optimistic": "Оптимистичный",
    "inv.scenario_conservative_sub": "нижняя граница 90%",
    "inv.scenario_base_sub": "прогноз модели",
    "inv.scenario_optimistic_sub": "верхняя граница 90%",
    "inv.scenarios_warning":
      "Базовый сценарий — основной прогноз. Крайние границы покрывают ~90% похожих квартир, но не гарантируют исход. Реальная арендная плата зависит от мебели, ремонта, сезона и переговорной силы — модель этого не знает.",
    "inv.s.rent": "Аренда",
    "inv.s.net_yield": "Чистая",
    "inv.s.payback": "Окуп.",
    "inv.s.npv": "NPV",

    // Verdicts
    "v.excellent": "Отлично",
    "v.good": "Хорошо",
    "v.average": "Средне",
    "v.questionable": "Сомнительно",
    "v.poor": "Плохо",

    // Errors
    "err.url_required": "Введите ссылку на krisha.kz.",
    "err.url_invalid": "Это не похоже на ссылку krisha.kz/a/show/…",
    "err.not_found": "Не удалось получить объявление. Возможно, оно удалено или скрыто продавцом.",
    "err.unavailable": "Сервер не отвечает. Проверьте, что бэкенд запущен на localhost:8000.",
    "err.bad_request": "Запрос отклонён: {detail}",
    "err.unknown": "Что-то пошло не так: {detail}",
    "err.prompt_short": "Опишите пожелания подробнее (минимум 3 символа).",
    "err.fallback_notice": "Примечание: ",

    // Loader
    "load.1": "🤔 Модель думает...",
    "load.2": "⏳ Ещё чуть-чуть, скоро ответим...",
    "load.3": "🧮 Считаем числа...",
    "load.4": "🌬️ Проверяем качество воздуха в районе...",
    "load.5": "🏙️ Смотрим что есть рядом...",
    "load.6": "🤖 ИИ почти готов...",
  },
  en: {
    "app.name": "AlmatyNest Intelligence",
    "app.tagline": "Almaty · AI-powered real estate",
    "nav.back": "← Back",
    "nav.start": "Get started",
    "nav.submit": "Analyse",
    "footer.disclaimer":
      "All estimates are guidance only, not financial advice. Models can be wrong — always verify independently.",
    "warning.freshness":
      "⚠ All models were trained on data up to March 2026. Predictions may not reflect recent market shifts.",

    // Landing
    "landing.subtitle":
      "Evaluate apartments, find a match, and decide whether a rental investment is worth it — three models on one site.",
    "landing.cta": "Get started",
    "landing.f1.title": "Fair price",
    "landing.f1.desc": "Check whether you're overpaying compared to the market.",
    "landing.f2.title": "Smart matching",
    "landing.f2.desc": "Describe what you want — the model finds matching apartments from thousands of listings.",
    "landing.f3.title": "Rental ROI",
    "landing.f3.desc": "Payback, NPV, IRR — should you buy this apartment to rent it out?",
    "landing.tech.title": "Under the hood",
    "landing.tech.desc": "LightGBM · conformal intervals · Gemini AI · MAPE 7.25% (sale), 12.3% (rent) · 89+ tests",
    "select.title": "Choose a model",
    "select.subtitle": "Each solves a different problem — click a card to open.",
    "m1.title": "Apartment evaluation",
    "m1.desc":
      "Paste a krisha.kz link — the model tells you if the listing is overpriced and compares the neighborhood.",
    "m2.title": "Recommendations",
    "m2.desc":
      "Describe your dream apartment in plain language — get 5–10 listings matched to your preferences.",
    "m3.title": "For investors",
    "m3.desc":
      "Get yield, payback period, and NPV for a listing, assuming you plan to rent it out.",
    "card.open": "Open →",

    // Evaluate
    "ev.title": "Apartment evaluation",
    "ev.url_label": "krisha.kz listing URL",
    "ev.url_placeholder": "https://krisha.kz/a/show/1009196093",
    "ev.use_llm": "Gemini response",
    "ev.use_llm_help":
      "When on, evaluation text is generated by Gemini AI. Falls back to template automatically if Gemini is overloaded.",
    "ev.r.predicted": "Model estimate",
    "ev.r.listed": "Listed price",
    "ev.r.delta": "Gap",
    "ev.r.interval": "Fair market price range",
    "ev.r.interval_tooltip":
      "Range in which the price is considered fair by the market. If the listing falls inside, you're not overpaying.",
    "ev.r.pros": "Pros",
    "ev.r.cons": "Cons",
    "ev.r.persona": "By persona",
    "ev.r.verdict": "Final verdict",
    "ev.r.link": "Listing examined",

    // Recommend
    "rc.title": "Recommendations",
    "rc.prompt_label": "Describe your dream apartment",
    "rc.prompt_placeholder":
      "E.g.: 2-room apartment in a complex under 40M near the center, not ground floor, with a school and park nearby.",
    "rc.params": "Advanced parameters",
    "rc.limit": "Number of results",
    "rc.limit_help": "How many listings to return.",
    "rc.mmr": "Diversity ↔ Relevance",
    "rc.mmr_help":
      "Closer to 1.0 — tighter match to your query. Closer to 0.0 — more diverse picks.",
    "rc.air": "Prioritize clean air",
    "rc.air_help":
      "Among equally good candidates, prefer listings with lower PM2.5.",
    "rc.use_llm": "Gemini response",
    "rc.use_llm_help": "When on, Gemini writes a short explanation per listing.",
    "rc.submit": "Find apartments",
    "rc.unmapped": "The model could not map the following wishes (rephrase to be explicit): ",
    "rc.summary_title": "Set overview",
    "rc.matched": "Listings passing the filter",
    "rc.score": "Match",
    "rc.price": "Price",
    "rc.area": "Area",
    "rc.rooms": "Rooms",
    "rc.district": "District",
    "rc.open": "Open on krisha.kz →",

    // Investment
    "inv.title": "Investment analysis",
    "inv.url_label": "krisha.kz listing URL",
    "inv.url_placeholder": "https://krisha.kz/a/show/1009196093",
    "inv.params": "Calculation parameters",
    "inv.vacancy": "Vacancy",
    "inv.vacancy_help":
      "Share of the year the apartment stands empty between tenants. For example, 0.08 means roughly 1 month idle per year.",
    "inv.repair": "Pre-rental repair",
    "inv.repair_help":
      "One-time cosmetic and furniture costs before the first rental. Expressed as a share of the sale price.",
    "inv.agent": "Agent fee",
    "inv.agent_help":
      "How many months of rent you pay the agent for each new tenant. In Almaty this is typically 0.5–1 month.",
    "inv.turnover": "Average tenancy length",
    "inv.turnover_help":
      "How many years on average a single tenant stays before moving out. Drives how often you pay the agent fee.",
    "inv.maintenance": "Maintenance",
    "inv.maintenance_help":
      "Ongoing costs: small repairs, appliance replacement, utility bills during vacancy. Share of annual rent.",
    "inv.tax": "Property tax",
    "inv.tax_help":
      "Annual property tax as a share of sale price. Kazakhstan rate for individuals is 0.1–1% depending on price.",
    "inv.inflation": "Inflation",
    "inv.inflation_help":
      "Expected annual CPI inflation in percent. Used to compute real payback and the discount rate.",
    "inv.risk": "Risk premium",
    "inv.risk_help":
      "Extra return in percentage points above inflation required for illiquid real estate. Discount rate = inflation + risk premium.",
    "inv.horizon": "Investment horizon",
    "inv.horizon_help": "Number of years for NPV (net present value) calculation.",
    "inv.use_llm": "Gemini response",
    "inv.use_llm_help": "When on, Gemini writes a full analytical investment report.",
    "inv.submit": "Calculate",
    "inv.r.lead": "Headline",
    "inv.r.price": "Price & model estimate",
    "inv.r.invest": "Investment numbers",
    "inv.r.market": "Market context",
    "inv.r.location": "Location for rent",
    "inv.r.risks": "Risks",
    "inv.r.verdict": "Final verdict",
    "inv.r.link": "Listing examined",
    "inv.r.thresholds_title": "What does this verdict mean?",
    "inv.r.thresholds_intro":
      "The verdict is based on inflation-adjusted payback (real payback, NPV method):",
    "inv.r.threshold_excellent": "Excellent — payback < 8 years",
    "inv.r.threshold_good": "Good — 8–12 years",
    "inv.r.threshold_average": "Average — 12–16 years",
    "inv.r.threshold_poor": "Poor — > 16 years (or gross yield < 70% of market)",
    "inv.r.threshold_current": "Current real payback:",
    "inv.r.thresholds_context":
      "In Kazakhstan 12.3% inflation + 3 pp risk premium yields a ≈15% discount rate. At that rate real payback tends to be long, so most apartments fall into 'average' or 'poor'. To reach 'good' you need gross yield of 10–12%+, a lower risk premium (if you accept more risk), or a cheaper entry price.",
    "inv.scenarios_title": "Rent scenarios range",
    "inv.scenarios_lead":
      "How the verdict shifts if achievable rent ends up below or above the model's prediction. These bounds are the 90% confidence interval from the training data.",
    "inv.scenario_conservative": "Conservative",
    "inv.scenario_base": "Base",
    "inv.scenario_optimistic": "Optimistic",
    "inv.scenario_conservative_sub": "lower 90% bound",
    "inv.scenario_base_sub": "model prediction",
    "inv.scenario_optimistic_sub": "upper 90% bound",
    "inv.scenarios_warning":
      "The base scenario is the main prediction. The bounds cover ~90% of similar listings, not a guarantee. Actual rent depends on furniture, renovation, season, and negotiation — the model can't see those.",
    "inv.s.rent": "Rent",
    "inv.s.net_yield": "Net",
    "inv.s.payback": "Payback",
    "inv.s.npv": "NPV",

    // Verdicts
    "v.excellent": "Excellent",
    "v.good": "Good",
    "v.average": "Average",
    "v.questionable": "Questionable",
    "v.poor": "Poor",

    // Errors
    "err.url_required": "Enter a krisha.kz URL.",
    "err.url_invalid": "This doesn't look like a krisha.kz/a/show/… URL.",
    "err.not_found": "Could not fetch the listing. It may be deleted or hidden by the seller.",
    "err.unavailable": "Server not responding. Make sure the backend runs on localhost:8000.",
    "err.bad_request": "Request rejected: {detail}",
    "err.unknown": "Something went wrong: {detail}",
    "err.prompt_short": "Describe your wishes in more detail (3+ characters).",
    "err.fallback_notice": "Notice: ",

    // Loader
    "load.1": "🤔 The model is thinking...",
    "load.2": "⏳ Almost there...",
    "load.3": "🧮 Crunching numbers...",
    "load.4": "🌬️ Checking air quality in the area...",
    "load.5": "🏙️ Looking around the neighborhood...",
    "load.6": "🤖 AI is almost done...",
  },
};

export function t(lang: Lang, key: string, vars?: Record<string, string>): string {
  const bucket = T[lang] ?? T.ru;
  let raw = bucket[key] ?? T.ru[key] ?? key;
  if (vars) {
    Object.entries(vars).forEach(([k, v]) => { raw = raw.replace(`{${k}}`, v); });
  }
  return raw;
}

export const LOADER_KEYS = ["load.1","load.2","load.3","load.4","load.5","load.6"] as const;
