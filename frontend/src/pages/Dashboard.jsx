import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { ChevronDown, ChevronUp } from "lucide-react";
import Papa from "papaparse";

// Existing chart components (adjust paths as needed)
import EmotionsStackedBarChart from "../components/dashboard_components/EmotionsStackedBarChart";
import MultiStudentMotionLineChart from "../components/dashboard_components/MultiStudentMotionLineChart";

// If using a card-based student summary approach
import StudentSummaryCards from "../components/dashboard_components/StudentSummaryCards";

function Dashboard() {
  const location = useLocation();
  const navigate = useNavigate();

  // If user came from Analysis, you might have a video title:
  const videoTitle = location.state?.videoTitle || "Classroom Analysis";

  // CSV data states
  const [emotionData, setEmotionData] = useState([]);
  const [motionData, setMotionData] = useState([]);
  const [loading, setLoading] = useState(true);

  // Collapsible sections
  const [showVideo, setShowVideo] = useState(true);
  const [showCharts, setShowCharts] = useState(true);
  const [showSummary, setShowSummary] = useState(true);

  // Only show the video if the user specifically uploaded "zoom_1_min.mp4"
  const isZoomVideo = videoTitle === "zoom_1_min.mp4";

  useEffect(() => {
    const fetchCSVs = async () => {
      try {
        // Make sure these CSV files exist in "public/logs/"
        const [emotionResponse, motionResponse] = await Promise.all([
          fetch("/logs/real_time_emotion_log.csv"),
          fetch("/logs/real_time_motion_log.csv"),
        ]);

        const emotionCSV = await emotionResponse.text();
        const motionCSV = await motionResponse.text();

        // Parse with Papa
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

  // Navigate to "/report" when "Create Report" button is clicked
  const handleCreateReport = () => {
    // Optionally pass default data via route state, e.g.:
    // navigate("/report", { state: { title: videoTitle, subject: "Some Subject" } });
    navigate("/report");
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
                    <video className="w-full max-w-3xl mx-auto rounded-lg shadow-lg" controls src="/zoom.mp4" />
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
                    <h2 className="text-xl font-semibold text-gray-800 mb-4">Emotion Distribution by Frame</h2>
                    <EmotionsStackedBarChart emotionData={emotionData} />
                  </div>

                  {/* Motion Chart */}
                  <div className="bg-white shadow-lg rounded-lg p-6">
                    <h2 className="text-xl font-semibold text-gray-800 mb-4">Cumulative Movement (All Students)</h2>
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
                  <h2 className="text-2xl text-center font-bold text-gray-200 mb-4">Student Summaries</h2>
                  <StudentSummaryCards motionData={motionData} emotionData={emotionData} />
                </div>
              )}
            </div>
          </>
        )}

        {/* Button to create a new Report */}
        <div className="flex mt-8 justify-center items-center">
          <button
            onClick={() => navigate("/report")}
            className="w-1/3 bg-blue-600 hover:bg-blue-700 text-xl text-white font-bold py-4 px-4 rounded-lg transition shadow-lg"
          >
            Create Report
          </button>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
