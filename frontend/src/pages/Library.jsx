import React, { useState } from "react";
import sessionData from "../data/Sessions.json";

const Library = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [filter, setFilter] = useState("");

  const sessions = sessionData.sessions;

  const filteredSessions = Object.values(sessions).filter(session =>
    session.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    session.subject.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="bg-gray-900 text-white min-h-screen p-6 flex flex-col items-center">
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg w-full max-w-4xl text-center mb-6">
        <h2 className="text-4xl font-bold text-white">Library</h2>
      </div>
      
      <div className="w-full max-w-4xl flex items-center justify-between bg-gray-800 p-4 rounded-lg shadow-md">
        <input
          type="text"
          placeholder="Search..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full p-2 bg-gray-700 text-white rounded-md mr-4"
        />
        <button 
          className="bg-white hover:bg-gray-200 text-black font-semibold p-3 rounded-md transition"
        >
          Search
        </button>
      </div>
      
      <div className="w-full max-w-4xl mt-6">
        {filteredSessions.length > 0 ? (
          filteredSessions.map((session, index) => (
            <div key={index} className="bg-gray-800 p-4 rounded-lg shadow-md mb-4">
              <h3 className="text-2xl font-bold text-white">{session.title}</h3>
              <p className="text-gray-300">Subject: {session.subject}</p>
              <p className="text-gray-400">Date: {session.date} | Time: {session.time}</p>
            </div>
          ))
        ) : (
          <p className="text-gray-300 text-center mt-4">No matching results found.</p>
        )}
      </div>
    </div>
  );
};

export default Library;
