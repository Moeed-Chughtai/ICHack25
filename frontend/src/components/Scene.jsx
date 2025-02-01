import React, { useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";

// New Component to Handle Rotation
const RotatingCap = () => {
  const capRef = useRef();
  const { scene } = useGLTF("/models/cap2.glb");

  // Rotate the cap slowly
  useFrame(() => {
    if (capRef.current) {
      capRef.current.rotation.y += 0.01; // Adjust speed if needed
    }
  });

  return <primitive ref={capRef} object={scene} scale={3} position={[0, 1, 0]} />;
};

const Scene = () => {
  return (
    <Canvas
      className="absolute top-[-50px] left-1/2 transform -translate-x-1/2"
      style={{ pointerEvents: "none" }} // Prevents blocking interactions with text
      gl={{ alpha: true }} // Makes the background transparent
      camera={{ position: [0, 2, 5], fov: 50 }}
    >
      {/* Camera Controls */}
      <OrbitControls enableZoom={false} enablePan={false} />

      {/* Lighting */}
      <ambientLight intensity={1.5} />
      <directionalLight position={[3, 5, 2]} intensity={1.5} />

      {/* 3D Model with Rotation */}
      <RotatingCap />
    </Canvas>
  );
};

export default Scene;
