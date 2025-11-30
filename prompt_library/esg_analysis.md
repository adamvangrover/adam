# Guide to ESG Analysis using the Prompt Library

## Introduction

This guide is designed to help you leverage our comprehensive JSON prompt library to conduct thorough and standardized ESG analysis. The goal of this library is to provide a structured framework for your analysis, ensuring all critical aspects of ESG analysis are considered consistently and efficiently.

## Overview of the Prompt Library JSON Structure

The provided JSON file is the backbone of your analysis. It's organized into several key sections:

* **`prompt_metadata`**: Contains general information about the prompt library version and author.
* **`core_analysis_areas`**: This is the heart of the library. It's an array of individual prompt objects, each designed to tackle a specific part of the ESG analysis process. Each prompt has an `id`, `title`, `description`, and a list of `prompts` that you can use to generate the analysis.

Your main focus will be on the `core_analysis_areas`, as these provide the building blocks for your ESG analysis reports.

## How to Use This Guide

This document will walk you through the typical workflow of an ESG analysis process. Each step in the process corresponds to a specific section of a standard ESG analysis report. For each step, this guide will:

1.  **Identify the relevant prompt(s)** from the library by its `prompt_title` and `(prompt_id)`.
2.  **Summarize the objective** of that analytical section.
3.  **List key questions** you should answer, based on the `prompts` in the JSON file, to build your analysis.

Think of this guide as a roadmap and the prompt library as your toolkit.

## Step-by-Step ESG Analysis Walkthrough

### I. ESG Investment Opportunity Scan

* **Objective**: To identify and analyze investment opportunities related to specific Environmental, Social, and Governance (ESG) themes or UN Sustainable Development Goals (SDGs).
* **Relevant Prompt(s) from Library**: ESG Investment Opportunity Scan (`esg_investment_opportunity_scan`)
