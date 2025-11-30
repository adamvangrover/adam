import React from 'react';
import News from '../components/news-and-insights/News';
import AdamsInsights from '../components/news-and-insights/AdamsInsights';
import LegalUpdates from '../components/news-and-insights/LegalUpdates';

const NewsAndInsights: React.FC = () => {
  return (
    <div>
      <h1>News and Insights</h1>
      <News />
      <AdamsInsights />
      <LegalUpdates />
    </div>
  );
};

export default NewsAndInsights;
