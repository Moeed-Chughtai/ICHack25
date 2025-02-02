// MultiStudentMotionLineChart.jsx
import React, { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

const MultiStudentMotionLineChart = ({ motionData }) => {
  // Transform data: group by frame so that each object has the frame plus each student’s cumulative movement.
  const chartData = useMemo(() => {
    const frameMap = {};
    motionData.forEach((row) => {
      const frame = row.frame;
      const student = row.student_id;
      const cumulative = parseFloat(row.cumulative_movement) || 0;
      if (!frameMap[frame]) {
        frameMap[frame] = { frame: Number(frame) };
      }
      // Use a key like "student_1", "student_2", etc.
      frameMap[frame][`student_${student}`] = cumulative;
    });
    return Object.values(frameMap).sort((a, b) => a.frame - b.frame);
  }, [motionData]);

  // Get a list of distinct student IDs.
  const students = useMemo(() => {
    return Array.from(new Set(motionData.map((row) => row.student_id)));
  }, [motionData]);

  // A set of colors for the lines.
  const colors = [
    "#8884d8",
    "#82ca9d",
    "#ff7300",
    "#ff0000",
    "#00ff00",
    "#0000ff",
    "#ffff00",
    "#ff00ff",
    "#00ffff",
  ];

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="frame" />
        <YAxis />
        <Tooltip />
        <Legend />
        {students.map((student, index) => (
          <Line
            key={student}
            type="monotone"
            dataKey={`student_${student}`}
            stroke={colors[index % colors.length]}
            strokeWidth={2}
            dot={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
};

export default MultiStudentMotionLineChart;
