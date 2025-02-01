import React from "react";
import {useNavigate} from "react-router-dom";

const Home = () => {

  const navigate = useNavigate();

  return (
    <div className="bg-gray-900 text-white min-h-screen">
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
          <p className="mt-6 inline-block bg-blue-500 hover:bg-blue-600 text-white text-lg px-6 py-3 rounded-full shadow-md transition duration-300" onClick={() => navigate("/analysis")}>
            Get Started
          </p>
        </div>
      </header>

      {/* Features Section */}
      <section id="features" className="py-20 px-6 bg-gray-800">
        <div className="max-w-6xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-white mb-6">Our Features</h2>
          <p className="text-gray-300 mb-10">Automated classroom and </p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="p-6 bg-gray-900 rounded-lg shadow-md hover:scale-105 transition">
              <h3 className="text-2xl font-semibold text-blue-400">🚀 </h3>
              <p className="mt-2 text-gray-400">Real time analuysis</p>
            </div>

            <div className="p-6 bg-gray-900 rounded-lg shadow-md hover:scale-105 transition">
              <h3 className="text-2xl font-semibold text-green-400">🔐 </h3>
              <p className="mt-2 text-gray-400">text</p>
            </div>

            <div className="p-6 bg-gray-900 rounded-lg shadow-md hover:scale-105 transition">
              <h3 className="text-2xl font-semibold text-yellow-400">🌎 AI-Powered</h3>
              <p className="mt-2 text-gray-400">Automated analysis of youe teaching and student needs.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-16 px-6 text-center bg-blue-600">
        <h2 className="text-4xl font-bold text-white">Footer</h2>
        <p className="mt-3 text-gray-100">More footer stuff</p>
        <a href="/signup" className="mt-5 inline-block bg-white text-blue-600 font-semibold px-6 py-3 rounded-full shadow-md hover:bg-gray-200 transition">
          blabla
        </a>
      </section>

      
    </div>
  );
};

export default Home;
