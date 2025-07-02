package com.narrativelibrary.model;

import java.util.List;
import java.util.Map;
import java.util.Date;

public class Company {
    private String id;
    private String name;
    private String industryId;
    private String ownershipStructure;
    private String corporateStructure;
    private Map<String, Object> financials;
    private List<String> companySpecificDriverIds;
    private TradingLevelData tradingLevels;

    public Company() {
    }

    public Company(String id, String name, String industryId, String ownershipStructure, String corporateStructure, Map<String, Object> financials, List<String> companySpecificDriverIds, TradingLevelData tradingLevels) {
        this.id = id;
        this.name = name;
        this.industryId = industryId;
        this.ownershipStructure = ownershipStructure;
        this.corporateStructure = corporateStructure;
        this.financials = financials;
        this.companySpecificDriverIds = companySpecificDriverIds;
        this.tradingLevels = tradingLevels;
    }

    // Getters
    public String getId() { return id; }
    public String getName() { return name; }
    public String getIndustryId() { return industryId; }
    public String getOwnershipStructure() { return ownershipStructure; }
    public String getCorporateStructure() { return corporateStructure; }
    public Map<String, Object> getFinancials() { return financials; }
    public List<String> getCompanySpecificDriverIds() { return companySpecificDriverIds; }
    public TradingLevelData getTradingLevels() { return tradingLevels; }

    // Setters
    public void setId(String id) { this.id = id; }
    public void setName(String name) { this.name = name; }
    public void setIndustryId(String industryId) { this.industryId = industryId; }
    public void setOwnershipStructure(String ownershipStructure) { this.ownershipStructure = ownershipStructure; }
    public void setCorporateStructure(String corporateStructure) { this.corporateStructure = corporateStructure; }
    public void setFinancials(Map<String, Object> financials) { this.financials = financials; }
    public void setCompanySpecificDriverIds(List<String> companySpecificDriverIds) { this.companySpecificDriverIds = companySpecificDriverIds; }
    public void setTradingLevels(TradingLevelData tradingLevels) { this.tradingLevels = tradingLevels; }
}

// Define other POJOs conceptually here or in separate files
// e.g., Industry.java, Driver.java, TradingLevelData.java, NarrativeExplanation.java, MacroEnvironmentFactor.java

class TradingLevelData {
    private Double price;
    private Long volume;
    private Double volatility;
    private Date timestamp;

    // Constructors, Getters, Setters
    public Double getPrice() { return price; }
    public void setPrice(Double price) { this.price = price; }
    public Long getVolume() { return volume; }
    public void setVolume(Long volume) { this.volume = volume; }
    public Double getVolatility() { return volatility; }
    public void setVolatility(Double volatility) { this.volatility = volatility; }
    public Date getTimestamp() { return timestamp; }
    public void setTimestamp(Date timestamp) { this.timestamp = timestamp; }
}
