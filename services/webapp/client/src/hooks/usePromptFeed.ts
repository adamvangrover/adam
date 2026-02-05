import { useEffect, useRef } from 'react';
import { usePromptStore } from '../stores/promptStore';
import { calculatePromptAlpha } from '../utils/alphaCalculator';
import { generateSyntheticPrompt } from '../utils/simulationEngine';
import { PromptObject } from '../types/promptAlpha';

export function usePromptFeed() {
  const feeds = usePromptStore((state) => state.feeds);
  const preferences = usePromptStore((state) => state.preferences);
  const addPrompts = usePromptStore((state) => state.addPrompts);
  const updateFeedStatus = usePromptStore((state) => state.updateFeedStatus);

  const intervalsRef = useRef<Record<string, NodeJS.Timeout>>({});
  const simIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Memoize feed configuration to prevent infinite loops when status updates
  const feedsConfigKey = JSON.stringify(feeds.map(f => ({ id: f.id, url: f.url, isActive: f.isActive })));

  useEffect(() => {
    const fetchFeed = async (feedId: string, url: string) => {
      updateFeedStatus(feedId, 'loading');
      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();

        const newPrompts: PromptObject[] = [];

        // Parse Reddit JSON
        if (data && data.data && Array.isArray(data.data.children)) {
          data.data.children.forEach((child: any) => {
            const post = child.data;
            const content = post.selftext || '';

            // Basic filter: must have content
            if (content.length < 50) return;

            const metrics = calculatePromptAlpha(content);

            // Alpha Threshold Filter
            if (metrics.score < preferences.minAlphaThreshold) return;

            newPrompts.push({
              id: `reddit-${post.id}`,
              title: post.title,
              content: content,
              source: 'Reddit',
              timestamp: post.created_utc * 1000,
              author: post.author,
              alphaScore: metrics.score,
              metrics: {
                length: metrics.lengthScore,
                variableDensity: metrics.variableScore,
                structuralKeywords: metrics.structureScore,
              },
              tags: [post.subreddit_name_prefixed],
              isFavorite: false,
            });
          });
        }

        if (newPrompts.length > 0) {
          addPrompts(newPrompts);
        }
        updateFeedStatus(feedId, 'success');

      } catch (error) {
        console.error(`Failed to fetch feed ${feedId}:`, error);
        updateFeedStatus(feedId, 'error');
      }
    };

    // Start feeds
    feeds.forEach(feed => {
      if (!feed.isActive) return;

      // Initial fetch
      fetchFeed(feed.id, feed.url);

      // Interval fetch
      if (intervalsRef.current[feed.id]) clearInterval(intervalsRef.current[feed.id]);

      intervalsRef.current[feed.id] = setInterval(() => {
        fetchFeed(feed.id, feed.url);
      }, preferences.refreshInterval);
    });

    return () => {
      // Cleanup
      Object.values(intervalsRef.current).forEach(clearInterval);
      intervalsRef.current = {};
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [feedsConfigKey, preferences.refreshInterval, preferences.minAlphaThreshold, addPrompts, updateFeedStatus]);


  // Separate Effect for Simulation
  useEffect(() => {
    if (simIntervalRef.current) {
        clearInterval(simIntervalRef.current);
        simIntervalRef.current = null;
    }

    if (preferences.useSimulation) {
        // Initial synthetic burst
        const initialBatch = Array.from({ length: 5 }, () => generateSyntheticPrompt());
        addPrompts(initialBatch);

        // Ongoing drip
        simIntervalRef.current = setInterval(() => {
            if (Math.random() > 0.3) { // 70% chance to generate per tick
                addPrompts([generateSyntheticPrompt()]);
            }
        }, 5000); // Check every 5s
    }

    return () => {
        if (simIntervalRef.current) {
            clearInterval(simIntervalRef.current);
        }
    };
  }, [preferences.useSimulation, addPrompts]);

  return { feeds };
}
