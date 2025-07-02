package com.narrativelibrary.model;

import java.util.List;

public class Industry {
    private String id;
    private String name;
    private String description;
    private List<String> macroDriverIds;
    private List<String> industrySpecificDriverIds;

    // Constructors, Getters, Setters
    public Industry() {}

    public Industry(String id, String name, String description, List<String> macroDriverIds, List<String> industrySpecificDriverIds) {
        this.id = id;
        this.name = name;
        this.description = description;
        this.macroDriverIds = macroDriverIds;
        this.industrySpecificDriverIds = industrySpecificDriverIds;
    }

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public List<String> getMacroDriverIds() { return macroDriverIds; }
    public void setMacroDriverIds(List<String> macroDriverIds) { this.macroDriverIds = macroDriverIds; }
    public List<String> getIndustrySpecificDriverIds() { return industrySpecificDriverIds; }
    public void setIndustrySpecificDriverIds(List<String> industrySpecificDriverIds) { this.industrySpecificDriverIds = industrySpecificDriverIds; }
}
