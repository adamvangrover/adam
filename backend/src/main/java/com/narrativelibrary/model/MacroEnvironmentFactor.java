package com.narrativelibrary.model;

public class MacroEnvironmentFactor {
    private String id;
    private String name;
    private Object currentValue; // Could be String or Number
    private String trend; // Consider Enum: Increasing, Decreasing, Stable
    private String impactNarrative;
    private String source;

    // Constructors, Getters, Setters
    public MacroEnvironmentFactor() {}

    public MacroEnvironmentFactor(String id, String name, Object currentValue, String trend, String impactNarrative, String source) {
        this.id = id;
        this.name = name;
        this.currentValue = currentValue;
        this.trend = trend;
        this.impactNarrative = impactNarrative;
        this.source = source;
    }

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public Object getCurrentValue() { return currentValue; }
    public void setCurrentValue(Object currentValue) { this.currentValue = currentValue; }
    public String getTrend() { return trend; }
    public void setTrend(String trend) { this.trend = trend; }
    public String getImpactNarrative() { return impactNarrative; }
    public void setImpactNarrative(String impactNarrative) { this.impactNarrative = impactNarrative; }
    public String getSource() { return source; }
    public void setSource(String source) { this.source = source; }
}
