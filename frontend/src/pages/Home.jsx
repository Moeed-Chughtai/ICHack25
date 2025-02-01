import React from "react";
import {useNavigate} from "react-router-dom";
import Scene from "../components/Scene";

const Home = () => {

  const navigate = useNavigate();

  return (
    <div className="bg-gray-900 text-white min-h-screen flex flex-col items-center justify-center">
      {/* 3D Cap Positioned Above the Title */}
      <div className="relative w-full h-[200px]">
        <Scene />
      </div>

      {/* Hero Section */}
      <header className="relative h-screen flex flex-col justify-center items-center text-center px-6">
        <div className="absolute inset-0 bg-cover bg-center opacity-40" style={{ backgroundImage: `url('https://source.unsplash.com/random/1920x1080?technology')` }}></div>
        
        <div className="relative z-10 max-w-3xl">
          <h1 className="text-5xl md:text-6xl font-extrabold text-white leading-tight">
            Welcome to <span className="text-blue-400">PreDictEd</span>
          </h1>
          <p className="mt-4 text-lg text-gray-300">
          Automated analysis of youe teaching and student needs.
          </p>
          <a href="#features" className="mt-6 inline-block bg-blue-500 hover:bg-blue-600 text-white text-lg px-6 py-3 rounded-full shadow-md transition duration-300">
            Get Started
          </a>
        </div>
      </header>
    </div>
  );
};

export default Home;
