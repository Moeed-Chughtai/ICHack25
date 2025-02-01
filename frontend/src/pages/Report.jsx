import React, { useState } from "react";
import { useLocation } from "react-router-dom";

const Report = () => {
  const location = useLocation();
  const initialData = location.state || { title: "", subject: "" }; // Default values

  const [formData, setFormData] = useState({
    title: initialData.title,
    subject: initialData.subject,
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Submitted Report:", formData);
    alert("Report Submitted Successfully!");
  };

  return (
    <div className="bg-white text-white min-h-screen p-6 flex flex-col items-center">
      <form onSubmit={handleSubmit} className="bg-gray-800 p-6 rounded-lg shadow-lg w-full max-w-4xl">
        <h2 className="text-4xl font-bold text-white mb-4 text-center">Report</h2>
        
        <div className="mb-4">
          <label className="block text-gray-300">Title</label>
          <input
            type="text"
            name="title"
            value={formData.title}
            onChange={handleChange}
            className="w-full p-2 bg-gray-700 text-white rounded-md"
            required
          />
        </div>

        <div className="mb-4">
          <label className="block text-gray-300">Subject</label>
          <input
            type="text"
            name="subject"
            value={formData.subject}
            onChange={handleChange}
            className="w-full p-2 bg-gray-700 text-white rounded-md"
            required
          />
        </div>

        <h2 className="text-2xl font-bold text-white mb-4">Movement Graph</h2>
        <h2 className="text-2xl font-bold text-white mb-4">Students</h2>

        <button
          type="submit"
          className="w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold p-3 rounded-md transition"
        >
          Submit Report
        </button>
      </form>
    </div>
  );
};

export default Report;
