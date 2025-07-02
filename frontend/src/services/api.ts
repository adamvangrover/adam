import { Company, Driver, NarrativeExplanation } from '../models/core'; // Assuming models are in this path

const API_BASE_URL = 'http://localhost:8000'; // Adjust if your backend runs elsewhere

export interface CompanySummary {
    id: string;
    name: string;
}

export interface CompanyExplanationResponse {
    company_id: string;
    company_name: string;
    num_drivers_found: number;
    drivers: Driver[]; // Backend returns full driver objects here
    narrative_summary: string;
}


async function handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: response.statusText }));
        throw new Error(errorData.detail || errorData.message || `API request failed with status ${response.status}`);
    }
    return response.json() as Promise<T>;
}

export async function fetchCompanies(): Promise<CompanySummary[]> {
    const response = await fetch(`${API_BASE_URL}/companies`);
    return handleResponse<CompanySummary[]>(response);
}

export async function fetchCompanyDetails(companyId: string): Promise<Company> {
    // This endpoint doesn't exist yet in the backend, we'd need one that returns full Company object.
    // For now, let's assume we will construct it from other calls or backend adds it.
    // As a placeholder, let's try to get the explanation which contains some company info
    // Or, we can fetch company drivers and try to get company node info from the KG via another endpoint if needed.
    // The current /companies/{company_id}/explanation has some info.
    // The /companies endpoint only gives {id, name}.
    // We'll simulate fetching more details or adjust later.

    // For now, let's just fetch the explanation as it has company_name
    // and then combine with drivers. A dedicated /companies/{id} endpoint would be better.
    const explanation = await fetchCompanyExplanation(companyId);
    const drivers = await fetchCompanyDrivers(companyId); // This will be the Driver[]

    // Mocking a Company object based on available data
    return {
        id: explanation.company_id,
        name: explanation.company_name,
        // These would need to be populated by a dedicated endpoint or enhanced explanation response
        industryId: "UNKNOWN", // Placeholder
        companySpecificDriverIds: drivers.map(d => d.id), // This isn't quite right, drivers are full objects
        // financials, tradingLevels etc. would also come from a dedicated company detail endpoint
    } as Company; // Type assertion as we are mocking parts of it
}


export async function fetchCompanyDrivers(companyId: string): Promise<Driver[]> {
    const response = await fetch(`${API_BASE_URL}/companies/${companyId}/drivers`);
    return handleResponse<Driver[]>(response);
}

export async function fetchCompanyExplanation(companyId: string): Promise<CompanyExplanationResponse> {
    const response = await fetch(`${API_BASE_URL}/companies/${companyId}/explanation`);
    return handleResponse<CompanyExplanationResponse>(response);
}

export async function fetchDriverDetails(driverId: string): Promise<Driver> {
    const response = await fetch(`${API_BASE_URL}/drivers/${driverId}`);
    return handleResponse<Driver>(response);
}

export async function reloadDataAdmin(): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE_URL}/admin/reload-data`, { method: 'POST' });
    return handleResponse<{ message: string }>(response);
}
