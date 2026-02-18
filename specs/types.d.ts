export interface FinancialMetrics {
    revenue: number;
    ebitda: number;
    net_income: number;
    total_debt: number;
    cash: number;
    net_debt?: number;
    leverage_ratio?: number;
}

export interface RAGSource {
    doc_id: string;
    chunk_id: string;
    page_number: number;
}

export interface CreditMemoSection {
    title: string;
    content: string;
    citations: RAGSource[];
}

export interface CreditMemo {
    borrower_name: string;
    ticker: string;
    sector: string;
    report_date: string; // ISO Date
    risk_score: number;
    executive_summary: string;
    sections: CreditMemoSection[];
    historical_financials: Record<string, any>[];
    dcf_analysis: Record<string, any>;
    pd_model: Record<string, any>;
}

export interface Report {
    id: string;
    title: string;
    date: string;
    type: "DAILY_BRIEFING" | "MARKET_PULSE";
    content_html: string;
    metadata: Record<string, any>;
}

export interface Agent {
    name: string;
    role: string;
    description: string;
    tools: string[];
    model: string;
}

export interface Prompt {
    name: string;
    template: string;
    variables: string[];
}
