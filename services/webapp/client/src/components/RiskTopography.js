import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';

const RiskSurface = ({ data }) => {
  // data is expected to be an array of objects { strike: number, maturity: number, volatility: number, risk_score: number }
  // For this demo, we'll generate a surface geometry.

  const geometry = useMemo(() => {
    // Create a plane geometry
    const width = 10;
    const height = 10;
    const segments = 20;
    const geo = new THREE.PlaneGeometry(width, height, segments, segments);

    // Modify vertices based on volatility (Z-axis)
    const positions = geo.attributes.position;

    // Simple simulation of volatility surface logic if no real data
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const y = positions.getY(i);

      // Calculate Z based on a simple volatility smile function simulation
      // Volatility is higher at edges (smile) and changes with maturity (y)
      const distance = Math.sqrt(x * x + y * y);
      const z = Math.sin(distance * 2) * 0.5 + (x*x)/20;

      positions.setZ(i, z);
    }

    geo.computeVertexNormals();
    return geo;
  }, []);

  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      <meshStandardMaterial side={THREE.DoubleSide} vertexColors={false} color="#06b6d4" wireframe={true} />
    </mesh>
  );
};

const RiskNode = ({ position, color, label }) => {
    return (
        <group position={position}>
            <mesh>
                <sphereGeometry args={[0.2, 32, 32]} />
                <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} />
            </mesh>
            <Text position={[0, 0.4, 0]} fontSize={0.2} color="white">
                {label}
            </Text>
        </group>
    )
}


const RiskTopography = ({ data }) => {
    // Default Simulated Data for the "Vertical Slice" if no props provided
    const defaultData = [
        { name: "TechCorp (Distressed)", headroom: 0.05, risk: "High", pos: [-2, 0, 1] }, // < 10% Headroom -> Red
        { name: "EnergyCo", headroom: 0.25, risk: "Low", pos: [2, 1, -1] },
        { name: "RetailGrp", headroom: 0.12, risk: "Medium", pos: [0, -0.5, 0] },
    ];

    const portfolioCompanies = data || defaultData;

  return (
    <div className="h-96 w-full glass-panel border border-cyan-900/30 rounded-lg overflow-hidden">
      <div className="absolute top-4 left-4 z-10 text-cyan-400 font-mono text-xs">
         RISK TOPOGRAPHY // VOLATILITY SURFACE // 3D VIEW
      </div>
      <Canvas camera={{ position: [5, 5, 5], fov: 60 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#06b6d4" />
        <OrbitControls />
        <gridHelper args={[20, 20, 0x1a2e3b, 0x0f172a]} />

        {/* Render Risk Nodes */}
        {portfolioCompanies.map((company, idx) => {
            const isCritical = company.headroom < 0.10;
            const color = isCritical ? "#ef4444" : (company.headroom < 0.20 ? "#f59e0b" : "#10b981");
            return (
                <RiskNode
                    key={idx}
                    position={company.pos}
                    color={color}
                    label={`${company.name}\n${(company.headroom * 100).toFixed(1)}%`}
                />
            );
        })}

        {/* Render a base surface to represent the market environment */}
         <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -2, 0]}>
            <planeGeometry args={[20, 20, 20, 20]} />
            <meshStandardMaterial wireframe color="#1e293b" transparent opacity={0.3} />
        </mesh>

      </Canvas>
    </div>
  );
};

export default RiskTopography;
