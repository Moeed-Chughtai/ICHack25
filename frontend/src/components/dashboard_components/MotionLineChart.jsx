// MotionLineChart.jsx
import React, { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const MotionLineChart = ({ motionData }) => {
  // Prepare data for chart
  const chartData = useMemo(() => {
    const frameMap = {};

    motionData.forEach((row) => {
      const frame = row.frame;
      const delta = parseFloat(row.movement_delta) || 0;

      if (!frameMap[frame]) {
        frameMap[frame] = { frame: Number(frame), totalMovement: 0 };
      }
      frameMap[frame].totalMovement += delta;
    });

    // Convert to array, sorted by frame
    const finalData = Object.values(frameMap).sort((a, b) => a.frame - b.frame);
    return finalData;
  }, [motionData]);

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="frame" />
        <YAxis />
        <Tooltip />
        <Line
          type="monotone"
          dataKey="totalMovement"
          stroke="#6366F1" // Indigo
          strokeWidth={3}
          dot={true}
          activeDot={{ r: 8 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default MotionLineChart;
