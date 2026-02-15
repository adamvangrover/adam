
/**
 * CryptoArbitrageWidget.js
 *
 * Simulates the output of the 'CryptoArbitrageAgent' (Protocol ARCHITECT_INFINITE Day 6).
 * In a production environment, this would fetch real-time JSON from the Python backend.
 */

class CryptoArbitrageWidget {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.maxItems = 10;
        this.interval = null;

        // Mock Exchanges
        this.exchanges = ['Binance', 'Kraken', 'Coinbase', 'KuCoin', 'Bitfinex'];
        // Mock Pairs
        this.pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT'];

        this.init();
    }

    init() {
        if (!this.container) {
            console.error("CryptoArbitrageWidget: Container not found");
            return;
        }

        // Create Header
        const header = document.createElement('div');
        header.className = "flex items-center justify-between border-b border-cyan-500/30 pb-2 mb-2";
        header.innerHTML = `
            <div class="flex items-center gap-2">
                <i class="fas fa-bolt text-yellow-400 animate-pulse"></i>
                <span class="text-cyan-400 font-bold tracking-wider text-sm">ARBITRAGE SCANNER [LIVE]</span>
            </div>
            <div class="text-xs text-gray-500 font-mono" id="arb-status">SCANNING...</div>
        `;
        this.container.appendChild(header);

        // Create List Container
        this.list = document.createElement('div');
        this.list.className = "space-y-2 overflow-y-auto max-h-64 scrollbar-thin scrollbar-thumb-cyan-700 scrollbar-track-transparent pr-2";
        this.container.appendChild(this.list);

        // Start scanning simulation
        this.scan();
        this.interval = setInterval(() => this.scan(), 3000);
    }

    scan() {
        // 30% chance to find an opportunity
        if (Math.random() > 0.3) {
            const opp = this.generateOpportunity();
            this.addOpportunity(opp);

            // Update status
            const status = document.getElementById('arb-status');
            if (status) {
                status.innerText = `FOUND: ${opp.symbol}`;
                status.className = "text-xs text-green-400 font-mono animate-pulse";
                setTimeout(() => {
                    status.innerText = "SCANNING...";
                    status.className = "text-xs text-gray-500 font-mono";
                }, 1000);
            }
        }
    }

    generateOpportunity() {
        const pair = this.pairs[Math.floor(Math.random() * this.pairs.length)];
        const ex1 = this.exchanges[Math.floor(Math.random() * this.exchanges.length)];
        let ex2 = this.exchanges[Math.floor(Math.random() * this.exchanges.length)];
        while (ex1 === ex2) {
            ex2 = this.exchanges[Math.floor(Math.random() * this.exchanges.length)];
        }

        const basePrice = (pair.includes('BTC') ? 50000 : (pair.includes('ETH') ? 3000 : 100));
        const variance = (Math.random() * 0.05); // 0-5% spread
        const buyPrice = basePrice;
        const sellPrice = basePrice * (1 + variance);
        const profit = (sellPrice - buyPrice);
        const spread = (variance * 100).toFixed(2);

        return {
            symbol: pair,
            buyExchange: ex1,
            sellExchange: ex2,
            spread: spread,
            profit: profit.toFixed(2)
        };
    }

    addOpportunity(opp) {
        const item = document.createElement('div');
        item.className = "bg-gray-900/80 border-l-2 border-green-500 p-2 text-xs font-mono hover:bg-gray-800 transition-colors cursor-pointer group";

        // Color code spread
        const spreadColor = parseFloat(opp.spread) > 2.0 ? 'text-red-400' : 'text-green-400';

        item.innerHTML = `
            <div class="flex justify-between items-center mb-1">
                <span class="text-white font-bold">${opp.symbol}</span>
                <span class="${spreadColor} font-bold">+${opp.spread}%</span>
            </div>
            <div class="flex justify-between text-gray-400 text-[10px]">
                <span>BUY: ${opp.buyExchange}</span>
                <i class="fas fa-arrow-right text-gray-600"></i>
                <span>SELL: ${opp.sellExchange}</span>
            </div>
            <div class="mt-1 text-right text-yellow-500/80 opacity-0 group-hover:opacity-100 transition-opacity">
                EST. PROFIT: $${opp.profit}
            </div>
        `;

        // Prepend to list
        if (this.list.firstChild) {
            this.list.insertBefore(item, this.list.firstChild);
        } else {
            this.list.appendChild(item);
        }

        // Limit items
        if (this.list.children.length > this.maxItems) {
            this.list.removeChild(this.list.lastChild);
        }
    }
}

// Auto-initialize if container exists
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('crypto-arbitrage-widget')) {
        new CryptoArbitrageWidget('crypto-arbitrage-widget');
    }
});
