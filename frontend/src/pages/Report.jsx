import React, { useState } from "react";

const Report = () => {
  const [formData, setFormData] = useState({title: "", time: "", date: "", subject: "" });

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
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg w-full max-w-4xl text-center mb-10">
        <h2 className="text-4xl font-bold text-white mb-4">Report</h2>

      </div>

      {/* Report Input Section */}
      <form onSubmit={handleSubmit} className="bg-gray-800 p-6 rounded-lg shadow-lg w-full max-w-4xl">
        <h2 className="text-2xl font-bold text-white mb-4 text-center">Submit Report</h2>
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
