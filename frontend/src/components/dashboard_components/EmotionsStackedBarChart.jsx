// EmotionsStackedBarChart.jsx
import React, { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const EmotionsStackedBarChart = ({ emotionData }) => {
  // Transform the raw emotionData into chart-friendly data
  const chartData = useMemo(() => {
    // 1. Collect all unique emotion types
    const allEmotions = [...new Set(emotionData.map((row) => row.emotion))];
    
    // 2. Group by frame
    const framesMap = {};

    emotionData.forEach((item) => {
      const frame = item.frame;
      if (!framesMap[frame]) {
        framesMap[frame] = { frame: Number(frame) };
        // Initialize all possible emotions at 0
        allEmotions.forEach((em) => {
          framesMap[frame][em] = 0;
        });
      }
      // Increment the count for this emotion
      framesMap[frame][item.emotion] += 1;
    });

    // 3. Convert to an array sorted by frame
    const finalData = Object.values(framesMap).sort((a, b) => a.frame - b.frame);
    return finalData;
  }, [emotionData]);

  // We'll pick some nice colors for each emotion
  // Adjust as needed for more or fewer emotions
  const emotionColors = {
    happy: "#FBBF24",      // Amber
    sad: "#3B82F6",        // Blue
    fear: "#EF4444",       // Red
    neutral: "#9CA3AF",    // Gray
    "not frontal": "#10B981",  // Green
  };

  // Figure out which emotions are actually present in the data
  const uniqueEmotions = Object.keys(chartData[0] || {}).filter(
    (key) => key !== "frame"
  );

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={chartData} stackOffset="expand">
        <XAxis dataKey="frame" />
        <YAxis />
        <Tooltip />
        <Legend />
        {uniqueEmotions.map((emotion) => (
          <Bar
            key={emotion}
            dataKey={emotion}
            stackId="emotions"
            fill={emotionColors[emotion] || "#8884d8"}
            name={emotion}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
};

export default EmotionsStackedBarChart;
