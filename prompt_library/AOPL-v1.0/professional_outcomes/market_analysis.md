# Guide to Market Analysis using the Prompt Library

## Introduction

This guide is designed to help you leverage our comprehensive JSON prompt library to conduct thorough and standardized market analysis. The goal of this library is to provide a structured framework for your analysis, ensuring all critical aspects of market analysis are considered consistently and efficiently.

## Overview of the Prompt Library JSON Structure

The provided JSON file is the backbone of your analysis. It's organized into several key sections:

* **`prompt_metadata`**: Contains general information about the prompt library version and author.
* **`core_analysis_areas`**: This is the heart of the library. It's an array of individual prompt objects, each designed to tackle a specific part of the market analysis process. Each prompt has an `id`, `title`, `description`, and a list of `prompts` that you can use to generate the analysis.

Your main focus will be on the `core_analysis_areas`, as these provide the building blocks for your market analysis reports.

## How to Use This Guide

This document will walk you through the typical workflow of a market analysis process. Each step in the process corresponds to a specific section of a standard market analysis report. For each step, this guide will:

1.  **Identify the relevant prompt(s)** from the library by its `prompt_title` and `(prompt_id)`.
2.  **Summarize the objective** of that analytical section.
3.  **List key questions** you should answer, based on the `prompts` in the JSON file, to build your analysis.

Think of this guide as a roadmap and the prompt library as your toolkit.

## Step-by-Step Market Analysis Walkthrough

### I. Daily Market Briefing

* **Objective**: To generate a concise daily market briefing summarizing key market movements, news, and upcoming events.
* **Relevant Prompt(s) from Library**: Daily Market Briefing (`daily_market_briefing`)

### II. Sector Deep Dive Report

* **Objective**: To generate a comprehensive deep-dive report on a specific industry sector.
* **Relevant Prompt(s) from Library**: Sector Deep Dive Report (`sector_deep_dive_report`)

### III. Geopolitical Risk Impact Assessment

* **Objective**: To generate an assessment of the potential impact of a specific geopolitical event or trend on given asset classes or regions.
* **Relevant Prompt(s) from Library**: Geopolitical Risk Impact Assessment (`geopolitical_risk_impact_assessment`)

### IV. Market Shock Scenario Analysis

* **Objective**: To analyze the potential impact of a specified market shock event on various asset classes, sectors, or a specific portfolio.
* **Relevant Prompt(s) from Library**: Market Shock Scenario Analysis (`market_shock_scenario_analysis`)

### V. Macroeconomic Themed Investment Strategy

* **Objective**: To generate an investment strategy based on a specific macroeconomic theme.
* **Relevant Prompt(s) from Library**: Macroeconomic Themed Investment Strategy (`macroeconomic_themed_investment_strategy`)
