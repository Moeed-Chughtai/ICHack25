import React from "react";

const About = () => {
  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-700">
      <div className="bg-gray-800 text-white p-8 rounded-lg shadow-lg w-full max-w-3xl">
        <h2 className="text-4xl font-bold text-center mb-6">About Us</h2>
        <p className="text-gray-300 text-center mb-6">
          We are a passionate team dedicated to providing the best solutions for classroom insights and analytics.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Mission Card */}
          <div className="bg-gray-700 p-6 rounded-lg shadow-md">
            <h3 className="text-xl font-bold text-white mb-2">Our Mission</h3>
            <p className="text-gray-400">
              Our goal is to enhance classroom experiences through real-time analysis and AI-powered insights.
            </p>
          </div>

          {/* Vision Card */}
          <div className="bg-gray-700 p-6 rounded-lg shadow-md">
            <h3 className="text-xl font-bold text-white mb-2">Our Vision</h3>
            <p className="text-gray-400">
              We aim to bridge the gap between technology and education, ensuring an engaging and effective learning environment.
            </p>
          </div>
        </div>

        {/* Call to Action */}
        <p className="text-center text-gray-400 mt-8">
          Want to know more? <a href="/contact" className="text-blue-400 hover:underline">Contact us</a>.
        </p>
      </div>
    </div>
  );
};

export default About;
