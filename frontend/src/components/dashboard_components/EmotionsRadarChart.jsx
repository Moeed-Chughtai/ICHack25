// EmotionsRadarChart.jsx
import React, { useState, useMemo } from "react";
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const EmotionsRadarChart = ({ emotionData }) => {
  // Get unique student IDs from the emotion data.
  const students = useMemo(() => {
    return Array.from(new Set(emotionData.map((row) => row.student_id)));
  }, [emotionData]);

  const [selectedStudent, setSelectedStudent] = useState(students[0] || "");

  // Aggregate the total duration for each emotion for the selected student.
  const radarData = useMemo(() => {
    if (!selectedStudent) return [];
    const emotionMap = {};
    emotionData.forEach((row) => {
      if (row.student_id === selectedStudent) {
        const emotion = row.emotion;
        const duration = parseFloat(row.emotion_duration) || 0;
        if (!emotionMap[emotion]) emotionMap[emotion] = 0;
        emotionMap[emotion] += duration;
      }
    });
    return Object.entries(emotionMap).map(([emotion, duration]) => ({
      emotion,
      duration,
    }));
  }, [emotionData, selectedStudent]);

  return (
    <div>
      <div className="mb-4">
        <label htmlFor="studentSelect" className="mr-2 font-bold">
          Select Student:
        </label>
        <select
          id="studentSelect"
          value={selectedStudent}
          onChange={(e) => setSelectedStudent(e.target.value)}
          className="border rounded p-1"
        >
          {students.map((student) => (
            <option key={student} value={student}>
              {student}
            </option>
          ))}
        </select>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={radarData}>
          <PolarGrid />
          <PolarAngleAxis dataKey="emotion" />
          <PolarRadiusAxis />
          <Tooltip />
          <Radar
            name="Emotion Duration"
            dataKey="duration"
            stroke="#8884d8"
            fill="#8884d8"
            fillOpacity={0.6}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default EmotionsRadarChart;
