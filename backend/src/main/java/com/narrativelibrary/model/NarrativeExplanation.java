package com.narrativelibrary.model;

import java.util.Date;
import java.util.List;

public class NarrativeExplanation {
    private String id;
    private String generatedForEntityId;
    private String entityType; // "Company" or "Industry"
    private List<String> driverIds;
    private String explanationText;
    private String linkedTradingLevelContext;
    private Date timestamp;

    // Constructors, Getters, Setters
    public NarrativeExplanation() {}

    public NarrativeExplanation(String id, String generatedForEntityId, String entityType, List<String> driverIds, String explanationText, String linkedTradingLevelContext, Date timestamp) {
        this.id = id;
        this.generatedForEntityId = generatedForEntityId;
        this.entityType = entityType;
        this.driverIds = driverIds;
        this.explanationText = explanationText;
        this.linkedTradingLevelContext = linkedTradingLevelContext;
        this.timestamp = timestamp;
    }

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getGeneratedForEntityId() { return generatedForEntityId; }
    public void setGeneratedForEntityId(String generatedForEntityId) { this.generatedForEntityId = generatedForEntityId; }
    public String getEntityType() { return entityType; }
    public void setEntityType(String entityType) { this.entityType = entityType; }
    public List<String> getDriverIds() { return driverIds; }
    public void setDriverIds(List<String> driverIds) { this.driverIds = driverIds; }
    public String getExplanationText() { return explanationText; }
    public void setExplanationText(String explanationText) { this.explanationText = explanationText; }
    public String getLinkedTradingLevelContext() { return linkedTradingLevelContext; }
    public void setLinkedTradingLevelContext(String linkedTradingLevelContext) { this.linkedTradingLevelContext = linkedTradingLevelContext; }
    public Date getTimestamp() { return timestamp; }
    public void setTimestamp(Date timestamp) { this.timestamp = timestamp; }
}
