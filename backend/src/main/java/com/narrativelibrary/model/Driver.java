package com.narrativelibrary.model;

import java.util.List;
import java.util.Map;

public class Driver {
    private String id;
    private String name;
    private String description;
    private String type; // Consider Enum: Macroeconomic, Fundamental, Technical, Geopolitical, CompanySpecific, IndustrySpecific
    private String impactPotential; // Consider Enum: High, Medium, Low
    private String timeHorizon; // Consider Enum: Short-term, Medium-term, Long-term
    private Map<String, String> metrics;
    private List<String> relatedMacroFactorIds;

    // Constructors, Getters, Setters
    public Driver() {}

    public Driver(String id, String name, String description, String type, String impactPotential, String timeHorizon, Map<String, String> metrics, List<String> relatedMacroFactorIds) {
        this.id = id;
        this.name = name;
        this.description = description;
        this.type = type;
        this.impactPotential = impactPotential;
        this.timeHorizon = timeHorizon;
        this.metrics = metrics;
        this.relatedMacroFactorIds = relatedMacroFactorIds;
    }

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    public String getImpactPotential() { return impactPotential; }
    public void setImpactPotential(String impactPotential) { this.impactPotential = impactPotential; }
    public String getTimeHorizon() { return timeHorizon; }
    public void setTimeHorizon(String timeHorizon) { this.timeHorizon = timeHorizon; }
    public Map<String, String> getMetrics() { return metrics; }
    public void setMetrics(Map<String, String> metrics) { this.metrics = metrics; }
    public List<String> getRelatedMacroFactorIds() { return relatedMacroFactorIds; }
    public void setRelatedMacroFactorIds(List<String> relatedMacroFactorIds) { this.relatedMacroFactorIds = relatedMacroFactorIds; }
}
