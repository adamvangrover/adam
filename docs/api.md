## docs/api.md

## Adam v19.0 API Documentation

This document provides comprehensive details about the Adam v19.0 API, enabling seamless integration with various applications and services.

### Introduction

The Adam v19.0 API empowers developers to access a wide array of functionalities, including:

* Real-time and historical market data retrieval
* Comprehensive sentiment analysis (asset-specific and overall market)
* In-depth fundamental analysis (company data, valuations, and financial health)
* Advanced technical analysis (indicators, trading signals, and chart patterns)
* Sophisticated risk assessment (investment-specific and portfolio-wide)
* Portfolio management (construction, optimization, and performance tracking)
* Automated report generation (customizable and comprehensive)
* Access to the knowledge graph (querying and updating)
* Simulation execution (running various financial simulations)

### Authentication

All API requests require authentication via an API key. To obtain your unique key, please visit the Adam v19.0 platform and sign up for an account.

Include your API key in the `Authorization` header of your requests:

```
Authorization: Bearer YOUR_API_KEY
```

### Endpoints

#### Market Data

* **GET /market-data/{symbol}**: Retrieves current market data for the specified symbol (e.g., AAPL, GOOG, BTC-USD).

    * Parameters:
        * `symbol`: String representing the asset symbol.
    * Response:
        ```json
        {
          "symbol": "AAPL",
          "price": 170.34,
          "volume": 1000000,
          "market_cap": 2800000000,
          "change_percent": 1.2,
          # ... other relevant market data fields
        }
        ```

* **GET /market-data/history/{symbol}**: Retrieves historical market data for the specified symbol.

    * Parameters:
        * `symbol`: String
        * `start_date`: String (optional) in YYYY-MM-DD format.
        * `end_date`: String (optional) in YYYY-MM-DD format.
        * `interval`: String (optional) specifying the time interval (e.g., "1d", "1wk", "1mo").
    * Response:
        ```json
        {
          "historical_data": [
            {
              "date": "2023-03-01",
              "open": 165.00,
              "high": 168.50,
              "low": 164.20,
              "close": 167.80,
              "volume": 1200000
            },
            # ... other historical data points
          ]
        }
        ```

#### Sentiment Analysis

* **GET /sentiment/{asset}**: Analyzes market sentiment for the specified asset.

    * Parameters:
        * `asset`: String representing the asset (e.g., AAPL, gold, BTC).
    * Response:
        ```json
        {
          "asset": "AAPL",
          "sentiment_score": 0.75,
          "sentiment_summary": "positive",
          "sentiment_breakdown": {
            "positive": 0.8,
            "negative": 0.1,
            "neutral": 0.1
          },
          "sources": [
            "news_articles",
            "social_media",
            "prediction_markets"
          ]
        }
        ```

* **GET /sentiment/overall**: Analyzes overall market sentiment.

    * Response:
        ```json
        {
          "sentiment_score": 0.6,
          "sentiment_summary": "moderately bullish",
          "sentiment_breakdown": {
            "bullish": 0.5,
            "bearish": 0.2,
            "neutral": 0.3
          },
          "sources": [
            "news_articles",
            "social_media",
            "prediction_markets"
          ]
        }
        ```

#### Fundamental Analysis

* **GET /fundamental/{company}**: Retrieves fundamental data for the specified company.

    * Parameters:
        * `company`: String representing the company name or ticker symbol.
    * Response:
        ```json
        {
          "company_name": "Apple Inc.",
          "ticker_symbol": "AAPL",
          "financial_statements": {
            "income_statement": {
              "revenue": 394328000000,
              "net_income": 99803000000,
              # ... other income statement items
            },
            "balance_sheet": {
              "total_assets": 381189000000,
              "total_liabilities": 287912000000,
              # ... other balance sheet items
            },
            "cash_flow_statement": {
              "operating_cash_flow": 111443000000,
              "free_cash_flow": 80674000000,
              # ... other cash flow statement items
            }
          },
          "key_metrics": {
            "revenue_growth": 0.08,
            "profit_margin": 0.25,
            "debt_to_equity": 1.98,
            # ... other relevant metrics
          }
        }
        ```

* **POST /fundamental/valuation**: Performs a company valuation based on provided data.

    * Request Body:
        ```json
        {
          "company_data": {
            # ... company information and financial statements
          },
          "valuation_method": "DCF",
          "discount_rate": 0.1,
          "terminal_growth_rate": 0.02,
          # ... other parameters specific to the valuation method
        }
        ```
    * Response:
        ```json
        {
          "valuation": 190.50,
          "valuation_method": "DCF",
          "valuation_details": {
            # ... details about the valuation calculation
          }
        }
        ```

#### Technical Analysis

* **GET /technical/{symbol}**: Retrieves technical indicators for the specified symbol.

    * Parameters:
        * `symbol`: String
        * `indicators`: Array of strings (e.g., ["SMA", "RSI", "MACD"])
        * `period`: Integer (optional) specifying the period for the indicators (e.g., 20, 50, 200)
    * Response:
        ```json
        {
          "symbol": "AAPL",
          "indicators": {
            "SMA_50": 165.20,
            "RSI_14": 60.5,
            "MACD": 2.3
          }
        }
        ```

* **POST /technical/signals**: Generates trading signals based on provided data.

    * Request Body:
        ```json
        {
          "price_data": [
            {
              "date": "2023-03-01",
              "open": 165.00,
              "high": 168.50,
              "low": 164.20,
              "close": 167.80,
              "volume": 1200000
            },
            # ... other historical data points
          ],
          "strategy": "moving_average_crossover",
          "short_period": 50,
          "long_period": 200
        }
        ```
    * Response:
        ```json
        {
          "signals": [
            {
              "timestamp": "2023-03-15T10:00:00Z",
              "signal": "buy"
            },
            # ... other signals
          ]
        }
        ```

#### Risk Assessment

* **POST /risk/assessment**: Assesses the risk associated with an investment based on provided data.

    * Request Body:
        ```json
        {
          "investment_data": {
            "asset_type": "stock",
            "symbol": "AAPL",
            "financial_data": {
              # ... financial data for the asset
            },
            "market_data": {
              # ... market data for the asset
            }
          }
        }
        ```
    * Response:
        ```json
        {
          "risk_score": 0.6,
          "risk_factors": {
            "market_risk": 0.2,
            "credit_risk": 0.1,
            "liquidity_risk": 0.1,
            "operational_risk": "low",
            "geopolitical_risk": "moderate",
            "industry_risk": "low"
          }
        }
        ```

#### Portfolio Management

* **GET /portfolio/{portfolio_id}**: Retrieves portfolio details.

    * Parameters:
        * `portfolio_id`: String representing the portfolio identifier.
    * Response:
        ```json
        {
          "portfolio_name": "My Portfolio",
          "holdings": [
            {
              "asset": "AAPL",
              "quantity": 100,
              "purchase_price": 150.00,
              "current_price": 170.34,
              # ... other relevant holding details
            },
            # ... other holdings
          ],
          "performance": {
            "total_value": 17034.00,
            "profit_loss": 2034.00,
            "return_percent": 0.1356
          }
        }
        ```

* **POST /portfolio/optimize**: Optimizes a portfolio based on provided parameters.

    * Request Body:
        ```json
        {
          "portfolio_data": {
            # ... current portfolio details
          },
          "optimization_criteria": "maximize_return",
          "constraints": {
            "risk_tolerance": "moderate",
            "investment_goals": "growth"
          }
        }
        ```
    * Response:
        ```json
        {
          "optimized_portfolio": {
            "holdings": [
              {
                "asset": "AAPL",
                "allocation": 0.3
              },
              # ... other holdings
            ],
            "performance_metrics": {
              "expected_return": 0.12,
              "risk": 0.18
            }
          }
        }
        ```

#### Report Generation

* **POST /report/generate**: Generates a customized report based on provided parameters.

    * Request Body:
        ```json
        {
          "report_type": "investment_analysis",
          "company_name": "Apple Inc.",
          "financial_data": {
            # ... financial data for the company
          },
          "market_data": {
            # ... market data for the company
          }
        }
        ```
    * Response:
        ```json
        {
          "report": "[Generated Report Content]"
        }
        ```

#### Knowledge Graph

* **GET /knowledge-graph/{entity_type}/{entity_name}**: Retrieves data for a specific entity from the knowledge graph.

    * Path Parameters:
        * `entity_type`: The type of entity (e.g., "company", "industry", "concept").
        * `entity_name`: The name of the entity.
    * Response:
        ```json
        {
          "entity_data": {
            "name": "Apple Inc.",
            "industry": "Technology",
            "financials": {
              "revenue": 394328000000,
              "net_income": 99803000000
            }
          }
        }
        ```

* **POST /knowledge-graph/update**: Updates the knowledge graph with new information.

    * Request Body:
        ```json
        {
          "entity_type": "company",
          "entity_name": "Apple Inc.",
          "data": {
            "ceo": "Tim Cook",
            # ... other data to update
          }
        }
        ```
    * Response:
        ```json
        {
          "status": "success",
          "message": "Knowledge graph updated successfully."
        }
        ```

#### Simulations

* **POST /simulations/{simulation_name}**: Runs a specified simulation.

    * Path Parameters:
        * `simulation_name`: The name of the simulation to run (e.g., "credit_rating_assessment", "investment_committee").
    * Request Body:
        ```json
        {
          "company_name": "Example Company",
          "financial_data": {
            "revenue": 1000000,
            "net_income": 100000,
            "total_assets": 5000000,
            "total_liabilities": 2000000
          },
          "investment_amount": 1000000,
          "investment_horizon": "5 years"
        }
        ```
    * Response:
        ```json
        {
          "simulation_results": {
            "decision": "Approve",
            "rationale": "The investment is approved based on the favorable analysis and moderate risk.",
            "report": "[Simulation Report]"
          }
        }
        ```

### Request and Response Formats

All API requests and responses utilize JSON format for seamless data exchange. Detailed specifications for each endpoint, including request parameters, response formats, and error codes, are provided in the respective sections above.

### Error Handling

The API uses standard HTTP status codes to indicate the success or failure of a request.

* `200 OK`: The request was successful.
* `400 Bad Request`: The request was invalid or malformed.
* `401 Unauthorized`: The API key is missing or invalid.
* `404 Not Found`: The requested resource was not found.
* `500 Internal Server Error`: An unexpected error occurred on the server.

### Rate Limiting

The API is subject to rate limiting to prevent abuse. The specific rate limits will be communicated in the response headers.

### Versioning

The API is versioned to ensure compatibility. The current version is `v1`. Future versions will be released with backward compatibility in mind.

### Support

For any questions or issues related to the API, please contact [email protected]
