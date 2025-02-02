// StudentSummaryCards.jsx

import React, { useMemo, useState } from "react";
import { createPortal } from "react-dom";
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const StudentSummaryCards = ({ motionData, emotionData }) => {
  const DISTRACTION_THRESHOLD = 3.0;
  const NEGATIVE_EMOTION_THRESHOLD = 0.2;
  const NEGATIVE_EMOTIONS = ["sad", "fear"];

  const studentSummaries = useMemo(() => {
    const studentMap = {};

    // Gather motion data
    motionData.forEach((row) => {
      const studentId = row.student_id;
      const cumulative = parseFloat(row.cumulative_movement) || 0;
      if (!studentMap[studentId]) {
        studentMap[studentId] = {
          studentId,
          maxCumulativeMovement: 0,
          emotionDurations: {},
        };
      }
      if (cumulative > studentMap[studentId].maxCumulativeMovement) {
        studentMap[studentId].maxCumulativeMovement = cumulative;
      }
    });

    // Gather emotion data
    emotionData.forEach((row) => {
      const studentId = row.student_id;
      const emotion = row.emotion;
      const duration = parseFloat(row.emotion_duration) || 0;
      if (!studentMap[studentId]) {
        studentMap[studentId] = {
          studentId,
          maxCumulativeMovement: 0,
          emotionDurations: {},
        };
      }
      if (!studentMap[studentId].emotionDurations[emotion]) {
        studentMap[studentId].emotionDurations[emotion] = 0;
      }
      studentMap[studentId].emotionDurations[emotion] += duration;
    });

    return Object.values(studentMap).map((student) => {
      let total = 0;
      let negativeTotal = 0;
      Object.entries(student.emotionDurations).forEach(([emo, dur]) => {
        total += dur;
        if (NEGATIVE_EMOTIONS.includes(emo)) {
          negativeTotal += dur;
        }
      });

      const negativeFraction = total > 0 ? negativeTotal / total : 0;

      return {
        ...student,
        isDistracted: student.maxCumulativeMovement > DISTRACTION_THRESHOLD,
        isNegative: negativeFraction >= NEGATIVE_EMOTION_THRESHOLD,
      };
    });
  }, [motionData, emotionData]);

  // State for controlling the popup
  const [selectedStudent, setSelectedStudent] = useState(null);

  // Prepare radar data for the selected student
  const radarData = useMemo(() => {
    if (!selectedStudent) return [];
    return Object.entries(selectedStudent.emotionDurations).map(
      ([emotion, duration]) => ({
        emotion,
        duration,
      })
    );
  }, [selectedStudent]);

  // Minimal modal with NO black overlay
  const Modal = ({ onClose, student }) => {
    return createPortal(
      <div
        className="fixed top-10 left-1/2 z-50 w-96 p-4 bg-white shadow-lg rounded
                   transform -translate-x-1/2 border border-gray-300"
      >
        <button
          className="absolute top-2 right-2 text-gray-500 hover:text-gray-800"
          onClick={onClose}
        >
          X
        </button>
        <h2 className="text-xl font-bold mb-2">
          Student {student.studentId}
        </h2>
        {student.isDistracted && (
          <div className="text-red-600 font-semibold mb-2">
            ⚠ High Movement: Potential Distraction!
          </div>
        )}
        {student.isNegative && (
          <div className="text-orange-600 font-semibold mb-2">
            ⚠ Negative Emotions Detected!
          </div>
        )}
        <p className="mb-4">
          <span className="font-bold">Cumulative Movement:</span>{" "}
          {student.maxCumulativeMovement.toFixed(2)}
        </p>

        {/* Radar Chart */}
        <div style={{ width: "100%", height: "250px" }}>
          <ResponsiveContainer>
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
      </div>,
      document.body // no dark overlay, just the popup
    );
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
      {studentSummaries.map((student) => {
        const flagged = student.isDistracted || student.isNegative;
        return (
          <div
            key={student.studentId}
            className={`p-4 rounded shadow cursor-pointer ${
              flagged ? "bg-red-50 border border-red-300" : "bg-white"
            }`}
            onClick={() => setSelectedStudent(student)}
          >
            <h3 className="text-lg font-bold mb-2">
              Student {student.studentId}
            </h3>
            {flagged && (
              <div className="text-red-600 font-semibold mb-1">
                ⚠ Needs Attention
              </div>
            )}
            <p>Cumulative: {student.maxCumulativeMovement.toFixed(2)}</p>
          </div>
        );
      })}

      {/* Show the popup only if a student is selected */}
      {selectedStudent && (
        <Modal
          student={selectedStudent}
          onClose={() => setSelectedStudent(null)}
        />
      )}
    </div>
  );
};

export default StudentSummaryCards;
