import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Report from './pages/Report'
import Analysis from './pages/Analysis'

import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
    
    <Router>
      <Navbar /> {/* Add a navbar for navigation */}
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/analysis" element={<Analysis />} />
        <Route path="/report" element={<Report />} />
        {/* <Route path="/" element={<Contact />} /> */}
      </Routes>
    </Router>

    </>
  )
}

export default App


