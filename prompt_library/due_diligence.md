# Guide to Due Diligence using the Prompt Library

## Introduction

This guide is designed to help you leverage our comprehensive JSON prompt library to conduct thorough and standardized due diligence on a company. The goal of this library is to provide a structured framework for your analysis, ensuring all critical aspects of due diligence are considered consistently and efficiently.

## Overview of the Prompt Library JSON Structure

The provided JSON file is the backbone of your analysis. It's organized into several key sections:

* **`prompt_metadata`**: Contains general information about the prompt library version and author.
* **`core_analysis_areas`**: This is the heart of the library. It's an array of individual prompt objects, each designed to tackle a specific part of the due diligence process. Each prompt has an `id`, `title`, `description`, `instructions`, and a crucial list of `key_considerations`.

Your main focus will be on the `core_analysis_areas`, as these provide the building blocks for your due diligence checklist.

## How to Use This Guide

This document will walk you through the typical workflow of a due diligence process. Each step in the process corresponds to a specific section of a standard due diligence checklist. For each step, this guide will:

1.  **Identify the relevant prompt(s)** from the library by its `prompt_title` and `(prompt_id)`.
2.  **Summarize the objective** of that analytical section.
3.  **List key questions** you should answer, based on the `key_considerations` in the prompt, to build your analysis.

Think of this guide as a roadmap and the prompt library as your toolkit.

## Step-by-Step Due Diligence Walkthrough

### I. Comprehensive Due Diligence Checklist

* **Objective**: To generate a comprehensive checklist of items and questions for conducting due diligence on a company, covering business, financial, legal, and management aspects.
* **Relevant Prompt(s) from Library**: Comprehensive Due Diligence Checklist (`comprehensive_due_diligence_checklist`)

### II. Financial Due Diligence

* **Objective**: To generate a detailed checklist for conducting financial due diligence on a company.
* **Relevant Prompt(s) from Library**: Financial Due Diligence (`financial_due_diligence`)

### III. Operational Due Diligence

* **Objective**: To generate a detailed checklist for conducting operational due diligence on a company.
* **Relevant Prompt(s) from Library**: Operational Due Diligence (`operational_due_diligence`)

### IV. Legal Due Diligence

* **Objective**: To generate a detailed checklist for conducting legal due diligence on a company.
* **Relevant Prompt(s) from Library**: Legal Due Diligence (`legal_due_diligence`)
