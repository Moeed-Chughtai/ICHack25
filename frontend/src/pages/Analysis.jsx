// Analysis.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const Analysis = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [videoSrc, setVideoSrc] = useState(null);
  const [fileName, setFileName] = useState("No file chosen");

  // Handle file selection
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && selectedFile.type.startsWith("video/")) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      const videoURL = URL.createObjectURL(selectedFile);
      setVideoSrc(videoURL);
    } else {
      alert("Please upload a valid MP4 video file.");
      setFile(null);
      setFileName("No file chosen");
      setVideoSrc(null);
    }
  };

  // Navigate to Dashboard to show charts
  const handleAnalyze = () => {
    if (!file) {
      alert("Please upload a video first");
      return;
    }

    navigate("/dashboard", { state: { videoTitle: file.name } });
  };

  return (
    <div className="bg-gray-700 text-white min-h-screen flex flex-col items-center p-6">
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg w-full max-w-4xl text-center">
        <h2 className="text-4xl font-bold text-white mb-6">Upload Video for Analysis</h2>

        <form className="flex flex-col items-center">
          <label className="w-full p-2 bg-gray-300 text-black rounded-md mb-4 cursor-pointer text-center">
            {fileName}
            <input
              type="file"
              accept="video/mp4"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>
        </form>

        {/* Video Preview (Centered) */}
        {videoSrc && (
          <div className="mt-6 flex justify-center">
            <div className="w-full max-w-lg">
              <h3 className="text-xl font-semibold text-white mb-2">Video Preview:</h3>
              <video controls src={videoSrc} className="w-full rounded-md shadow-lg" />
            </div>
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
