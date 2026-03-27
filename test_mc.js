const fs = require('fs');

const CONFIG = { futureSteps: 30, simulationsPerEntity: 20 };

class MonteCarloEngine {
    static calculateStats(prices) {
        let returns = [];
        for (let i = 1; i < prices.length; i++) {
            returns.push(Math.log(prices[i] / prices[i - 1]));
        }

        const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
        const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
        const stdDev = Math.sqrt(variance);

        return { drift: mean, vol: stdDev, lastPrice: prices[prices.length - 1] };
    }

    static run(prices, outlook, riskScore, useBias) {
        const stats = this.calculateStats(prices);
        let paths = [];

        let bias = 0;
        if (useBias) {
            if (outlook.conviction === 'High') bias += 0.002;
            if (outlook.consensus === 'Buy') bias += 0.001;
            if (outlook.consensus === 'Sell') bias -= 0.002;
            if (riskScore > 80) bias -= 0.003;
        }

        const drift = stats.drift + bias;
        const dt = 1;

        if (isNaN(drift) || isNaN(stats.vol) || isNaN(stats.lastPrice)) {
            console.log("NaN detected in stats!", {stats, bias, drift});
        }

        for (let s = 0; s < CONFIG.simulationsPerEntity; s++) {
            let path = [stats.lastPrice];
            let currentPrice = stats.lastPrice;

            for (let t = 1; t <= CONFIG.futureSteps; t++) {
                const shock = (Math.random() + Math.random() + Math.random() + Math.random() - 2) / 2;
                const change = currentPrice * (drift * dt + stats.vol * shock * Math.sqrt(dt));
                currentPrice += change;
                if (currentPrice < 0.01) currentPrice = 0.01;
                path.push(currentPrice);

                if (isNaN(currentPrice)) {
                    console.log("NaN in currentPrice!", {currentPrice, change, drift, vol: stats.vol, shock});
                }
            }
            paths.push(path);
        }

        let meanPath = [];
        for (let t = 0; t <= CONFIG.futureSteps; t++) {
            let sum = 0;
            paths.forEach(p => sum += p[t]);
            meanPath.push(sum / CONFIG.simulationsPerEntity);
            if (isNaN(meanPath[meanPath.length - 1])) {
                console.log("NaN in meanPath", {t, sum});
            }
        }

        return { paths, meanPath, stats };
    }
}

const data = JSON.parse(fs.readFileSync('showcase/data/sp500_market_data.json', 'utf8'));
for (const item of data) {
    // console.log("Processing", item.ticker);
    try {
        const mc = MonteCarloEngine.run(item.price_history, item.outlook, item.risk_score, true);
        const normFactor = 10 / item.price_history[0];
        if (isNaN(normFactor)) {
            console.log("NaN normFactor!", item.ticker);
        }
    } catch (e) {
        console.error("Error for", item.ticker, e);
    }
}
