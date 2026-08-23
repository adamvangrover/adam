export interface DataPoint {
  date: string;
  price: number;
  projected?: number;
}

// Helper to generate dates
const generateDates = (start: Date, days: number): string[] => {
  const dates = [];
  for (let i = 0; i < days; i++) {
    const d = new Date(start);
    d.setDate(start.getDate() + i);
    dates.push(d.toISOString().split('T')[0]);
  }
  return dates;
};

// Generates historical and projected data using a random walk
export const generateAssetData = (startPrice: number, volatility: number, daysHistorical = 60, daysProjected = 30): DataPoint[] => {
  const today = new Date();
  const startDate = new Date(today);
  startDate.setDate(today.getDate() - daysHistorical);

  const dates = generateDates(startDate, daysHistorical + daysProjected);
  const data: DataPoint[] = [];

  let currentPrice = startPrice;

  for (let i = 0; i < dates.length; i++) {
    const isProjected = i >= daysHistorical;

    // Random walk
    const change = 1 + (Math.random() * volatility * 2 - volatility);
    currentPrice = currentPrice * change;

    if (isProjected) {
      data.push({
        date: dates[i],
        price: null as unknown as number, // Not real historical
        projected: Number(currentPrice.toFixed(2))
      });
    } else {
       data.push({
        date: dates[i],
        price: Number(currentPrice.toFixed(2))
      });
    }
  }

  return data;
};

// Pre-generated mock datasets
export const stockData = generateAssetData(150, 0.02);
export const bondData = generateAssetData(100, 0.005);
export const etfData = generateAssetData(500, 0.015);
export const cryptoData = generateAssetData(60000, 0.04);