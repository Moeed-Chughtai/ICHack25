import React, { useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";

// Rotating Cap Component
const RotatingCap = () => {
  const capRef = useRef();
  const { scene } = useGLTF("/models/cap2.glb");

  // Slow rotation
  useFrame(() => {
    if (capRef.current) {
      capRef.current.rotation.y += 0.01;
    }
  });

  return <primitive ref={capRef} object={scene} scale={3} position={[0, -1, 0]} />;
};

const Scene = () => {
  return (
    <Canvas
      className="absolute top-0 left-1/2 transform -translate-x-1/2"
      style={{ pointerEvents: "none" }} // Prevents blocking interactions
      gl={{ alpha: true }} // Transparent background
      camera={{ position: [0, 2, 5], fov: 50 }}
    >
      {/* Camera Controls */}
      <OrbitControls enableZoom={false} enablePan={false} />

      {/* Lighting */}
      <ambientLight intensity={1.5} />
      <directionalLight position={[3, 5, 2]} intensity={1.5} />

      {/* Rotating 3D Cap */}
      <RotatingCap />
    </Canvas>
  );
};

export default Scene;
