import React from "react";
import { useNavigate } from "react-router-dom";
import Scene from "../components/Scene";

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="bg-gray-900 text-white min-h-screen flex flex-col items-center">
      {/* 3D Cap Positioned Above the Title */}
      <div className="relative w-full h-[150px] flex items-center justify-center">
        <Scene />
      </div>

      {/* Hero Section */}
      <header className="relative flex flex-col items-center text-center px-6 py-20">
        <div
          className="absolute inset-0 bg-cover bg-center opacity-40"
          style={{ backgroundImage: `url('https://source.unsplash.com/random/1920x1080?technology')` }}
        ></div>

        <div className="relative z-10 max-w-3xl">
          <h1 className="text-5xl md:text-6xl font-extrabold text-white leading-tight">
            Welcome to <span className="text-blue-400">ThinkSync</span>
          </h1>
          <p className="mt-4 text-lg text-gray-300">
            Automated analysis of teaching and student needs.
          </p>
          <p className="mt-6 inline-block bg-blue-500 hover:bg-blue-600 text-white text-lg px-6 py-3 rounded-full shadow-md transition duration-300" onClick={() => navigate("/analysis")}>
            Get Started
          </p>
        </div>
      </header>

      {/* Features Section */}
      <section id="features" className="py-20 px-6 bg-gray-800 w-full">
        <div className="max-w-6xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-white mb-6">Our Features</h2>
          {/* <p className="text-gray-300 mb-10">Automated classroom and student insights</p> */}

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <FeatureCard title="🚀 Real-time Analysis" description="Get real-time insights on classroom interactions." />
            <FeatureCard title="🔐 Audio Detection" description="Get insight on classroom interactions." />
            <FeatureCard title="🌎 AI-Powered" description="Automated analysis to enhance learning experiences." />
          </div>
        </div>
      </section>

      {/* Call to Action (Footer) */}
      <section className="py-16 px-6 text-center bg-blue-600 w-full">
        <h2 className="text-4xl font-bold text-white">Join Us</h2>
        <p className="mt-3 text-gray-100">Enhance teaching experience with AI-powered tools.</p>
        <a
          href="/signup"
          className="mt-5 inline-block bg-white text-blue-600 font-semibold px-6 py-3 rounded-full shadow-md hover:bg-gray-200 transition"
        >
          Sign Up Now
        </a>
      </section>
    </div>
  );
};

// Feature Card Component
const FeatureCard = ({ title, description }) => (
  <div className="p-6 bg-gray-900 rounded-lg shadow-md hover:scale-105 transition">
    <h3 className="text-2xl font-semibold text-blue-400">{title}</h3>
    <p className="mt-2 text-gray-400">{description}</p>
  </div>
);

export default Home;
