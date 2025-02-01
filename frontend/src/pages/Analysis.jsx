import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const Analysis = () => {
    const navigate = useNavigate();
    const [file, setFile] = useState(null);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleUpload = (event) => {
        event.preventDefault();
        if (file) {
            console.log("Uploaded File:", file.name);
            alert("File uploaded successfully!");
        } else {
            alert("Please select a file first");
        }
    };

    // function to handle when pressing analyze
    // it should run the video and audio analysis
    const handleAnalyze = () => {
        if (!file) {
            alert("Please upload a file first");
            return;
        }
    
        const reportData = {
            title: file.name, // Using file name as title
            subject: "Automated Analysis" // Example subject
        };
    
        navigate("/report", { state: reportData });
    };
    

    return (
        <div className="bg-gray-900 text-white min-h-screen flex flex-col items-center p-6">
            <div className="bg-gray-800 p-6 rounded-lg shadow-lg w-full max-w-4xl text-center">
                <h2 className="text-4xl font-bold text-white mb-6">Upload File for Analysis</h2>
                
                <form onSubmit={handleUpload} className="flex flex-col items-center">
                    <input 
                        type="file" 
                        onChange={handleFileChange} 
                        className="w-full p-2 bg-gray-700 text-white rounded-md mb-4"
                    />
                    <button 
                        type="submit" 
                        className="bg-blue-500 hover:bg-blue-600 text-white font-semibold p-3 rounded-md transition"
                    >
                        Upload File
                    </button>
                </form>
            </div>
            
            <button 
                className="mt-6 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded" 
                onClick={handleAnalyze}
            >
                Analyze
            </button>
        </div>
    );
};

export default Analysis;
