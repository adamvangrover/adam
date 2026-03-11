/**
 * Artisanal Tools Module
 * -----------------------------------------------------------------------------
 * Provides parsing and semantic-lite search capabilities for artisanal data.
 * Designed to be dynamically loaded by ModularDataManager.
 */

class ArtisanalTools {
    /**
     * Parses newline-delimited JSON (JSONL) text.
     * @param {string} text - The raw JSONL string.
     * @returns {Array} - Array of parsed objects.
     */
    static parseJSONL(text) {
        if (!text) return [];
        return text
            .split('\n')
            .map(line => line.trim())
            .filter(line => line.length > 0)
            .map(line => {
                try {
                    return JSON.parse(line);
                } catch (e) {
                    console.warn("[ArtisanalTools] Parse error for line:", line, e);
                    return null;
                }
            })
            .filter(item => item !== null);
    }

    /**
     * Performs a weighted keyword search ("Semantic-Lite") on the data.
     * @param {string} query - The search query.
     * @param {Array} data - The dataset to search (array of objects).
     * @returns {Array} - Ranked array of results.
     */
    static semanticSearch(query, data) {
        if (!query || !data) return [];

        const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);
        if (terms.length === 0) return [];

        const results = data.map(item => {
            let score = 0;
            const input = (item.input || "").toLowerCase();
            const output = (item.output || "").toLowerCase();

            // Exact phrase match boost
            if (input.includes(query.toLowerCase())) score += 10;
            if (output.includes(query.toLowerCase())) score += 5;

            // Term matching
            terms.forEach(term => {
                if (input.includes(term)) score += 3; // Higher weight for question/input
                if (output.includes(term)) score += 1; // Lower weight for answer/output
            });

            return { item, score };
        });

        // Filter and sort
        return results
            .filter(r => r.score > 0)
            .sort((a, b) => b.score - a.score)
            .map(r => r.item);
    }
}

// Expose globally
window.ArtisanalTools = ArtisanalTools;
