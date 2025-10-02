# Data Source Documentation: [Data Source Name]

## 1. Overview

*   **Provider:** What is the name of the data provider (e.g., Alpha Vantage, Reddit, Refinitiv)?
*   **Data Type:** What kind of data does this source provide (e.g., stock prices, social media sentiment, ESG scores)?
*   **Link to Documentation:** Provide a direct link to the official API documentation.

## 2. API Details

*   **Base URL:** The base URL for the API.
*   **Key Endpoints:** List the primary API endpoints used by the data source class.
    *   `GET /endpoint1`: Description of the endpoint.
    *   `POST /endpoint2`: Description of the endpoint.

## 3. Authentication

*   **Method:** How does the service authenticate requests (e.g., API Key, OAuth 2.0)?
*   **Configuration:** How are credentials managed? (e.g., via `config/api_keys.yaml`, environment variables).
*   **Setup Instructions:** Brief steps on how a user can obtain and configure the necessary credentials.

## 4. Data Schema

*   **Key Data Objects:** Describe the main data objects returned by the API.
*   **Example Response (JSON):** Provide a sample JSON response snippet for a key endpoint.
    ```json
    {
      "key": "value",
      "another_key": {}
    }
    ```

## 5. Rate Limiting & Usage Policies

*   **Rate Limits:** What are the API rate limits (e.g., requests per minute/day)?
*   **Usage Policies:** Are there any other terms of service or usage policies to be aware of?
*   **Error Handling:** How does the data source class handle hitting a rate limit?

## 6. How to Use

*   **Example Class Instantiation:**
    ```python
    from core.data_sources.your_data_source import YourDataSource

    data_source = YourDataSource(api_key="YOUR_API_KEY")
    data = data_source.fetch_data(some_parameter="value")
    ```
*   **Integration:** Briefly describe how this data source is intended to be used by agents.