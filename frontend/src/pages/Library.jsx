import React, { useState } from "react";
import { useNavigate } from "react-router-dom"; // Import for navigation
import sessionData from "../data/Sessions.json";

const Library = () => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedSession, setSelectedSession] = useState(null);

  // Sample "sessions" from your JSON
  const sessions = sessionData.sessions;

  // Filter sessions by search
  const filteredSessions = Object.values(sessions).filter((session) =>
    session.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    session.subject.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // When a user clicks on a session card
  const handleSessionClick = (session) => {
    setSelectedSession(session);
  };

  // Close the popup
  const closePopup = () => {
    setSelectedSession(null);
  };

  // Navigate to the Dashboard with the selected session's data
  const goToDashboard = () => {
    if (!selectedSession) return;
    // Pass session data in route state, so Dashboard can retrieve it via `location.state`
    navigate("/dashboard", { state: { session: selectedSession } });
  };

  return (
    <div className="bg-gray-700 text-white min-h-screen p-6 flex flex-col items-center">
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg w-full max-w-4xl text-center mb-6">
        <h2 className="text-4xl font-bold text-white">Library</h2>
      </div>

      {/* Search Box */}
      <div className="w-full max-w-4xl flex items-center justify-between bg-gray-800 p-4 rounded-lg shadow-md">
        <input
          type="text"
          placeholder="Search..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full p-2 bg-gray-300 text-black rounded-md mr-4"
        />
        <button className="bg-white hover:bg-gray-200 text-black font-semibold p-3 rounded-md transition">
          Search
        </button>
      </div>

      {/* Session List */}
      <div className="w-full max-w-4xl mt-6">
        {filteredSessions.length > 0 ? (
          filteredSessions.map((session, index) => (
            <div
              key={index}
              className="bg-gray-800 p-4 rounded-lg shadow-md mb-4 hover:cursor-pointer hover:bg-gray-600"
              onClick={() => handleSessionClick(session)}
            >
              <h3 className="text-2xl font-bold text-white">{session.title}</h3>
              <p className="text-gray-300">Subject: {session.subject}</p>
              <p className="text-gray-400">
                Date: {session.date} | Time: {session.time}
              </p>
            </div>
          ))
        ) : (
          <p className="text-gray-300 text-center mt-4">No matching results found.</p>
        )}
      </div>

      {/* Popup Modal */}
      {selectedSession && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg max-w-md text-center">
            <h3 className="text-2xl font-bold text-white">{selectedSession.title}</h3>
            <p className="text-gray-300">Subject: {selectedSession.subject}</p>
            <p className="text-gray-400">
              Date: {selectedSession.date} | Time: {selectedSession.time}
            </p>
            
            <div className="mt-4 space-x-3">
              {/* Go to Dashboard Button */}
              <button
                onClick={goToDashboard}
                className="bg-blue-500 hover:bg-blue-600 text-white font-semibold p-3 rounded-md transition"
              >
                Go to Dashboard
              </button>

              {/* Close Button */}
              <button
                onClick={closePopup}
                className="bg-red-500 hover:bg-red-600 text-white font-semibold p-3 rounded-md transition"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Library;
