import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { ChevronDown, ChevronUp } from "lucide-react";
import Papa from "papaparse";

import EmotionsStackedBarChart from "../components/dashboard_components/EmotionsStackedBarChart";
import MultiStudentMotionLineChart from "../components/dashboard_components/MultiStudentMotionLineChart";
import StudentSummaryCards from "../components/dashboard_components/StudentSummaryCards";

function Dashboard() {
  const location = useLocation();
  const navigate = useNavigate();

  const videoTitle = location.state?.videoTitle || "Classroom Analysis";
  const videoFile = location.state?.videoFile || null;

  // CSV data
  const [emotionData, setEmotionData] = useState([]);
  const [motionData, setMotionData] = useState([]);
  const [loading, setLoading] = useState(true);

  // Collapsible states
  const [showVideo, setShowVideo] = useState(true);
  const [showCharts, setShowCharts] = useState(true);
  const [showSummary, setShowSummary] = useState(true);

  // Only show the video if user specifically used "zoom_1_min.mp4"
  const isZoomVideo = videoTitle === "zoom_1_min.mp4";

  useEffect(() => {
    const fetchCSVs = async () => {
      try {
        const [emotionResponse, motionResponse] = await Promise.all([
          fetch("/logs/real_time_emotion_log.csv"),
          fetch("/logs/real_time_motion_log.csv"),
        ]);

        const emotionCSV = await emotionResponse.text();
        const motionCSV = await motionResponse.text();

        const parsedEmotion = Papa.parse(emotionCSV, {
          header: true,
          skipEmptyLines: true,
        }).data;
        const parsedMotion = Papa.parse(motionCSV, {
          header: true,
          skipEmptyLines: true,
        }).data;

        setEmotionData(parsedEmotion);
        setMotionData(parsedMotion);
      } catch (err) {
        console.error("Error fetching or parsing CSVs:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchCSVs();
  }, []);

  useEffect(() => {
    const analyzeVideo = async () => {
      if (!videoFile) {
        setAnalysisError("No video file provided.");
        return;
      }
  
      const formData = new FormData();
      formData.append("video", videoFile);
  
      try {
        const response = await fetch("http://localhost:5001/analyze-video", {
          method: "POST",
          body: formData,
        });
  
        const data = await response.json();
        if (data.success) {
          setAnalysisResults(data.results);
          setAnalysisError(null);
        } else {
          setAnalysisError(data.error || "Analysis failed");
        }
      } catch (err) {
        console.error("Network error:", err);
        setAnalysisError(err.message);
      }
    };
  
    analyzeVideo(); // ✅ Call function when component mounts
  }, [videoFile]); // ✅ Runs when `videoFile` is set
  

  // Handle "Create Report" button
  const handleCreateReport = () => {
    navigate("/report");
  };

  // <!-- NEW CODE: State for speech/analysis -->
  const [analysisResults, setAnalysisResults] = useState(null);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);

  // <!-- NEW CODE: Example function to POST video and retrieve results -->
  const handleAnalyzeVideo = async () => {
    try {
      if (!videoFile) {
        setAnalysisError("No video file provided.");
        return;
      }
  
      const formData = new FormData();
      formData.append("video", videoFile); 
  
      const response = await fetch("http://localhost:5001/analyze-video", {
        method: "POST",
        body: formData,
      });
  
      const data = await response.json();
      if (data.success) {
        setAnalysisResults(data.results); // { transcript, wpm, confidence, ... }
        setAnalysisError(null);
      } else {
        setAnalysisError(data.error || "Analysis failed");
      }
    } catch (err) {
      console.error("Network error:", err);
      setAnalysisError(err.message);
    }
  };
  

  return (
    <div className="min-h-screen bg-gray-700 text-white p-6">
      <div className="max-w-7xl mx-auto py-8">
        {/* Page Title */}
        <h1 className="text-3xl font-bold text-center mb-6">Classroom Insights</h1>

        {loading ? (
          <p className="text-lg font-semibold text-center">Loading data...</p>
        ) : (
          <>
            {/* VIDEO SECTION (Collapsible) */}
            {isZoomVideo && (
              <div className="mb-8">
                <button
                  onClick={() => setShowVideo(!showVideo)}
                  className="w-full bg-gray-500 hover:bg-gray-400 text-white font-bold py-3 px-6 rounded-lg flex justify-between items-center transition"
                >
                  {showVideo ? "Hide Video" : "Show Video"}
                  {showVideo ? <ChevronDown size={20} /> : <ChevronUp size={20} />}
                </button>
                {showVideo && (
                  <div className="mt-4">
                    <video
                      className="w-full max-w-3xl mx-auto rounded-lg shadow-lg"
                      controls
                      src="/zoom.mp4"
                    />
                  </div>
                )}
              </div>
            )}

            {/* CHARTS SECTION (Collapsible) */}
            <div className="mb-8">
              <button
                onClick={() => setShowCharts(!showCharts)}
                className="w-full bg-gray-600 hover:bg-gray-400 text-white font-bold py-3 px-6 rounded-lg flex justify-between items-center transition"
              >
                {showCharts ? "Hide Charts" : "Show Charts"}
                {showCharts ? <ChevronDown size={20} /> : <ChevronUp size={20} />}
              </button>
              {showCharts && (
                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Emotions Chart */}
                  <div className="bg-white shadow-lg rounded-lg p-6">
                    <h2 className="text-xl font-semibold text-gray-800 mb-4">
                      Emotion Distribution by Frame
                    </h2>
                    <EmotionsStackedBarChart emotionData={emotionData} />
                  </div>

                  {/* Motion Chart */}
                  <div className="bg-white shadow-lg rounded-lg p-6">
                    <h2 className="text-xl font-semibold text-gray-800 mb-4">
                      Cumulative Movement (All Students)
                    </h2>
                    <MultiStudentMotionLineChart motionData={motionData} />
                  </div>
                </div>
              )}
            </div>

            {/* STUDENT SUMMARY SECTION (Collapsible) */}
            <div className="mb-8">
              <button
                onClick={() => setShowSummary(!showSummary)}
                className="w-full bg-gray-600 hover:bg-gray-400 text-white font-bold py-3 px-6 rounded-lg flex justify-between items-center transition"
              >
                {showSummary ? "Hide Student Summary" : "Show Student Summary"}
                {showSummary ? <ChevronDown size={20} /> : <ChevronUp size={20} />}
              </button>
              {showSummary && (
                <div className="mt-6">
                  <h2 className="text-2xl text-center font-bold text-gray-200 mb-4">
                    Student Summaries
                  </h2>
                  <StudentSummaryCards
                    motionData={motionData}
                    emotionData={emotionData}
                  />
                </div>
              )}
            </div>

            {/* NEW CODE: SPEECH & CONFIDENCE ANALYSIS SECTION (Collapsible) */}
            <div className="mb-8">
              <button
                onClick={() => setShowAnalysis(!showAnalysis)}
                className="w-full bg-gray-600 hover:bg-gray-400 text-white font-bold py-3 px-6 rounded-lg flex justify-between items-center transition"
              >
                {showAnalysis ? "Hide Speech Analysis" : "Show Speech Analysis"}
                {showAnalysis ? <ChevronDown size={20} /> : <ChevronUp size={20} />}
              </button>

              {showAnalysis && (
                <div className="mt-6 bg-white text-black p-6 rounded-lg">
                  <h2 className="text-2xl font-bold mb-4">Speech Analysis Results</h2>

                  {/* Error Display */}
                  {analysisError && (
                    <p className="text-red-600 font-semibold mb-4">
                      {analysisError}
                    </p>
                  )}

                  {/* Show the results if available */}
                  {analysisResults && (
                    <>
                      <p><strong>Transcript:</strong> {analysisResults.transcript}</p>
                      <p><strong>Word Count:</strong> {analysisResults.word_count}</p>
                      <p><strong>Duration:</strong> {analysisResults.duration?.toFixed(2)} seconds</p>
                      <p><strong>Confidence:</strong> {analysisResults.confidence ? (analysisResults.confidence * 100).toFixed(2) + "%" : "N/A"}</p>
                      <p><strong>Words Per Minute (WPM):</strong> {analysisResults.wpm?.toFixed(2)}</p>
                      <p><strong>AI Analysis:</strong> {analysisResults.ai_analysis}</p>
                    </>
                  )}

                </div>
              )}
            </div>

            {/* Button to create a new Report */}
            <div className="flex mt-8 justify-center items-center">
              <button
                onClick={handleCreateReport}
                className="w-1/3 bg-blue-600 hover:bg-blue-700 text-xl text-white font-bold py-4 px-4 rounded-lg transition shadow-lg"
              >
                Create Report
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
