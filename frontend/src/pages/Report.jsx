import React, { useState } from "react";
import { useLocation , useNavigate } from "react-router-dom";
import sessionData from "../data/Sessions.json"; // Import the JSON data

const Report = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const initialData = location.state || { title: "", subject: "" }; // Default values
  
  const getCurrentDate = () => {
    const today = new Date();
    return today.toISOString().split("T")[0]; // Format: YYYY-MM-DD
  };

  const getCurrentTime = () => {
    const now = new Date();
    return now.toTimeString().split(" ")[0].slice(0, 5); // Format: HH:MM
  };
  
  const [formData, setFormData] = useState({
    title: initialData.title,
    subject: initialData.subject,
    date: getCurrentDate(),
    time: getCurrentTime()
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Add new report to the JSON data
    const newSessionId = Object.keys(sessionData.sessions).length + 1;
    sessionData.sessions[newSessionId] = formData;
    
    console.log("Updated Sessions:", sessionData);
    alert("Report Submitted Successfully!");
    navigate("/library");
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

        <div className="mb-4">
          <label className="block text-gray-300">Date</label>
          <input
            type="date"
            name="date"
            value={formData.date}
            onChange={handleChange}
            className="w-full p-2 bg-gray-700 text-white rounded-md"
            required
          />
        </div>

        <div className="mb-4">
          <label className="block text-gray-300">Time</label>
          <input
            type="time"
            name="time"
            value={formData.time}
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
