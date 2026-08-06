import { AdamNode } from "./core";

// ==============================================================================
// 3. Financial Universe / Entity Layer
// ==============================================================================

export interface FinancialEntity extends AdamNode {
  ticker?: string;
  name: string;
  assetClass:
    | "Equity"
    | "Bond"
    | "ETF"
    | "Index"
    | "Commodity"
    | "Currency"
    | "Derivative";
}

export interface Company extends FinancialEntity {}
export interface Government extends FinancialEntity {}
export interface Issuer extends FinancialEntity {}
export interface Exchange extends FinancialEntity {}
export interface Sector extends FinancialEntity {}
export interface Industry extends FinancialEntity {}
