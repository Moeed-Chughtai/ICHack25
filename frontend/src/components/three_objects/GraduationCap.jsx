import React from "react";
import { useGLTF } from "@react-three/drei";

const GraduationCap = () => {
  // Load the 3D model
  const { scene } = useGLTF("/models/cap.glb"); // Ensure your model path is correct

  // Click handler function
  const handleClick = () => {
    console.log("3D model clicked!");
  };

  return (
    
    <primitive 
      object={scene} 
      scale={1} 
      position={[0, -1, 0]} 
      onClick={handleClick} // Add onClick event
    />
  );
};

export default GraduationCap;
