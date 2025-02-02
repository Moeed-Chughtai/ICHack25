import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
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
    <div className="min-h-screen bg-gray-100 text-gray-900 p-4">
      <div className="max-w-7xl mx-auto py-8">
        {/* Page Title */}
        <h1 className="text-3xl font-bold mb-4">{videoTitle} - Classroom Insights</h1>

        {/* Button to create a new Report */}
        <button
          onClick={handleCreateReport}
          className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mb-4"
        >
          Create Report
        </button>

        {loading ? (
          <p>Loading data...</p>
        ) : (
          <>
            {/* VIDEO SECTION (Collapsible) */}
            {isZoomVideo && (
              <div className="mb-8">
                <button
                  onClick={() => setShowVideo(!showVideo)}
                  className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
                >
                  {showVideo ? "Hide Video" : "Show Video"}
                </button>
                {showVideo && (
                  <div className="mt-4">
                    <video
                      className="w-full max-w-2xl mx-auto rounded shadow"
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
                className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
              >
                {showCharts ? "Hide Charts" : "Show Charts"}
              </button>
              {showCharts && (
                <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-8">
                  {/* Emotions Chart */}
                  <div className="bg-white shadow p-4 rounded">
                    <h2 className="text-xl font-semibold mb-2">
                      Emotion Distribution by Frame
                    </h2>
                    <EmotionsStackedBarChart emotionData={emotionData} />
                  </div>

                  {/* Movement Chart */}
                  <div className="bg-white shadow p-4 rounded">
                    <h2 className="text-xl font-semibold mb-2">
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
                className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded"
              >
                {showSummary ? "Hide Student Summary" : "Show Student Summary"}
              </button>
              {showSummary && (
                <div className="mt-4">
                  <h2 className="text-2xl font-bold mb-4">Student Summaries</h2>
                  {/* Cards + popups for each student if you want them */}
                  <StudentSummaryCards
                    motionData={motionData}
                    emotionData={emotionData}
                  />
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
