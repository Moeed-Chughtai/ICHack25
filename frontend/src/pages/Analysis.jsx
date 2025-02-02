// Analysis.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const Analysis = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [videoSrc, setVideoSrc] = useState(null);

  // Handle file selection
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && selectedFile.type.startsWith("video/")) {
      setFile(selectedFile);
      const videoURL = URL.createObjectURL(selectedFile);
      setVideoSrc(videoURL);
    } else {
      alert("Please upload a valid MP4 video file.");
      setFile(null);
      setVideoSrc(null);
    }
  };

  // Simulate upload (or actually perform your backend upload)
  const handleUpload = async (event) => {
    event.preventDefault();
    if (file) {
      console.log("Uploaded File:", file.name);
      alert("File uploaded successfully!");
      // [Optional: do your actual backend upload or speech analysis here if needed]
    } else {
      alert("Please select a file first");
    }
  };

  // Navigate to Dashboard to show charts
  const handleAnalyze = () => {
    if (!file) {
      alert("Please upload a video first");
      return;
    }

    // Optionally, you can pass the file name or other data in route state
    navigate("/dashboard", { state: { videoTitle: file.name } });
  };

  return (
    <div className="bg-gray-900 text-white min-h-screen flex flex-col items-center p-6">
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg w-full max-w-4xl text-center">
        <h2 className="text-4xl font-bold text-white mb-6">Upload Video for Analysis</h2>

        <form onSubmit={handleUpload} className="flex flex-col items-center">
          <input
            type="file"
            accept="video/mp4"
            onChange={handleFileChange}
            className="w-full p-2 bg-gray-700 text-white rounded-md mb-4"
          />
          <button
            type="submit"
            className="bg-blue-500 hover:bg-blue-600 text-white font-semibold p-3 rounded-md transition"
          >
            Upload Video
          </button>
        </form>

        {/* Video Preview */}
        {videoSrc && (
          <div className="mt-6">
            <h3 className="text-xl font-semibold text-white mb-2">Video Preview:</h3>
            <video controls src={videoSrc} className="w-full max-w-lg rounded-md shadow-lg" />
          </div>
        )}
      </div>

      {/* Analyze Button */}
      <button
        className="mt-6 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        onClick={handleAnalyze}
        disabled={!file}
      >
        Analyze
      </button>
    </div>
  );
};

export default Analysis;
