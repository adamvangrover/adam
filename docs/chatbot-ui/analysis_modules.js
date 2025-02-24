// analysis_modules.js

function generateMarketSentimentAnalysis() {
    const latestNewsletter = newsletters[newsletters.length - 1];
    const marketSentiment = latestNewsletter.sections.find(section => section.title === "Market Mayhem (Executive Summary)");
    return marketSentiment.content;
}

function generateMacroeconomicAnalysis() {
    const latestNewsletter = newsletters[newsletters.length - 1];
    const macroAnalysis = latestNewsletter.sections.find(section => section.title === "Macroeconomic Analysis");
    if (macroAnalysis) {
        return macroAnalysis.content;
    } else {
        return "Macroeconomic analysis is not available in the current newsletter.";
    }
}

function generateGeopoliticalRiskAnalysis() {
    const latestNewsletter = newsletters[newsletters.length - 1];
    const geopoliticalRisks = latestNewsletter.sections.find(section => section.title === "Policy Impact & Geopolitical Outlook");
    if (geopoliticalRisks) {
        return geopoliticalRisks.content;
    } else {
        return "Geopolitical risk analysis is not available in the current newsletter.";
    }
}

function generateIndustryAnalysis(message) {
    const industries = ["Technology", "Healthcare", "Energy", "Financials", "Consumer Discretionary", "Consumer Staples", "Industrials", "Materials", "Utilities", "Real Estate", "Telecommunication Services"];
    let industry = industries[Math.floor(Math.random() * industries.length)];

    const match = message.match(/industry analysis for\s(.*)/i);
    if (match && match) {
        const requestedIndustry = match.trim();
        if (industries.includes(requestedIndustry)) {
            industry = requestedIndustry;
        } else {
            return `Sorry, I don't have analysis for ${requestedIndustry} yet. Try one of these: ${industries.join(', ')}`;
        }
    }

    // Fetch industry report
    const industryReport = industryReports[industry];
    if (industryReport) {
        let response = `Analysis for ${industry} sector:\n`;
        industryReport.sections.forEach(section => {
            response += `  ${section.title}:\n`;
            if (Array.isArray(section.content)) {
                section.content.forEach(item => {
                    response += `    - ${item}\n`;
                });
            } else {
                response += `    ${section.content}\n`;
            }
        });
        return response;
    } else {
        return `Sorry, I don't have a detailed report for ${industry} yet.`;
    }
}

function generateFundamentalAnalysis(message) {
    const companies = ["AAPL", "MSFT", "GOOG", "AMZN"];
    let company = companies[Math.floor(Math.random() * companies.length)];

    const match = message.match(/fundamental analysis for\s(.*)/i);
    if (match && match) {
        const requestedCompany = match.trim().toUpperCase();
        if (companies.includes(requestedCompany)) {
            company = requestedCompany;
        } else {
            return `Sorry, I don't have fundamental analysis for ${requestedCompany} yet. Try one of these: ${companies.join(', ')}`;
        }
    }

    const companyReport = companyReports[company];
    if (companyReport) {
        const financialPerformance = companyReport.sections.find(section => section.title === "Financial Performance");
        if (financialPerformance) {
            let response = `Fundamental analysis for ${company}:\n`;
            for (const metric in financialPerformance.metrics) {
                response += `  ${metric}:\n`;
                for (const year in financialPerformance.metrics[metric]) {
                    response += `    ${year}: ${financialPerformance.metrics[metric][year]}\n`;
                }
            }
            response += `\n${financialPerformance.analysis}`;
            return response;
        }
    }
    return `Sorry, I don't have detailed fundamental analysis for ${company} yet.`;
}

function generateTechnicalAnalysis(message) {
    //... (similar implementation as generateFundamentalAnalysis,
    //... but fetching data from the "Technical Analysis" section)
    const companies = ["AAPL", "MSFT", "GOOG", "AMZN"];
    let company = companies[Math.floor(Math.random() * companies.length)];

    const match = message.match(/technical analysis for\s(.*)/i);
    if (match && match) {
        const requestedCompany = match.trim().toUpperCase();
        if (companies.includes(requestedCompany)) {
            company = requestedCompany;
        } else {
            return `Sorry, I don't have technical analysis for ${requestedCompany} yet. Try one of these: ${companies.join(', ')}`;
        }
    }

    const companyReport = companyReports[company];
    if (companyReport) {
        const technicalAnalysis = companyReport.sections.find(section => section.title === "Technical Analysis");
        if (technicalAnalysis) {
            let response = `Technical analysis for ${company}:\n`;
            technicalAnalysis.indicators.forEach(indicator => {
                response += `  - ${indicator.name}: ${indicator.value} (${indicator.signal})\n`;
            });
            response += `\n${technicalAnalysis.analysis}`;
            return response;
        }
    }
    return `Sorry, I don't have detailed technical analysis for ${company} yet.`;
}

function generatePortfolioOptimization() {
    const riskTolerance = ["conservative", "moderate", "aggressive"][Math.floor(Math.random() * 3)];
    const expectedReturn = (Math.random() * 20).toFixed(2) + "%";
    return `Based on your ${riskTolerance} risk tolerance, your optimized portfolio is expected to generate a return of ${expectedReturn} over the next year.`;
}
