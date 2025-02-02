// StudentCumulativeMovementTable.jsx
import React, { useMemo } from 'react';

const StudentCumulativeMovementTable = ({ motionData }) => {
  const summary = useMemo(() => {
    const studentMap = {};
    motionData.forEach((row) => {
      const student = row.student_id;
      const cumulative = parseFloat(row.cumulative_movement) || 0;
      // Store the maximum cumulative value for the student.
      if (!studentMap[student] || cumulative > studentMap[student]) {
        studentMap[student] = cumulative;
      }
    });
    return Object.entries(studentMap).map(([student, cumulativeMovement]) => ({
      student,
      cumulativeMovement,
    }));
  }, [motionData]);

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Student ID
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Final Cumulative Movement
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {summary.map((row) => (
            <tr key={row.student}>
              <td className="px-6 py-4 whitespace-nowrap">{row.student}</td>
              <td className="px-6 py-4 whitespace-nowrap">
                {row.cumulativeMovement.toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default StudentCumulativeMovementTable;
