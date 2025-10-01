import React from 'react';
import { useTranslation } from 'react-i18next';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);


const TechnicalAnalysis = ({ data }) => {
  const { t } = useTranslation();
  if (!data || !data.price_history || !data.indicators) {
    return <p>Incompatible data format for technical analysis chart.</p>;
  }

  const { price_history, indicators } = data;
  const labels = price_history.map(item => item.date);

  const chartData = {
    labels,
    datasets: [
      {
        label: t('technicalAnalysis.price'),
        data: price_history.map(item => item.close),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
      },
      {
        label: `SMA (${indicators.moving_average.period})`,
        data: indicators.moving_average.values,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        borderDash: [5, 5],
      },
       {
        label: `RSI (${indicators.rsi.period})`,
        data: indicators.rsi.values,
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        yAxisID: 'y1'
      }
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: t('technicalAnalysis.title') + ` for ${data.ticker}`,
      },
    },
    scales: {
        y: {
            type: 'linear',
            display: true,
            position: 'left',
        },
        y1: {
            type: 'linear',
            display: true,
            position: 'right',
            grid: {
                drawOnChartArea: false, // only draw grid lines for the first Y axis
            },
        },
    },
  };

  return (
    <div className="Card">
      <h3>{t('technicalAnalysis.title')}</h3>
      <Line options={options} data={chartData} />
      <h4>{t('technicalAnalysis.summary')}</h4>
      <p>{indicators.summary}</p>
    </div>
  );
};

export default TechnicalAnalysis;
