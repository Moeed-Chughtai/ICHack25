// displays average movement and average sentiment
import React, { useState } from "react";

const SentimentChart = ({ sessionId }) => {
    const [chartUrl, setChartUrl] = useState("");

    const fetchSentimentChart = async () => {
        try {
            const response = await fetch(`http://127.0.0.1:5000/api/sentiment-chart?session_id=${sessionId}`);

            if (!response.ok) {
                throw new Error(`Error fetching chart: ${response.statusText}`);
            }

            const blob = await response.blob();
            setChartUrl(URL.createObjectURL(blob));
        } catch (error) {
            console.error("Error fetching sentiment chart:", error);
        }
    };

    return (
        <div>
            <button onClick={fetchSentimentChart}>Load Sentiment Chart</button>
            {chartUrl && <img src={chartUrl} alt="Sentiment Analysis Chart" style={{ width: 400, height: 400 }} />}
        </div>
    );
};

export default SentimentChart;
