import React, { useRef, useState, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, PerspectiveCamera, Environment, useGLTF, Center, Html } from '@react-three/drei';
import * as THREE from 'three';
import { Plus, Minus, RotateCcw, Play, Pause, Compass } from 'lucide-react';
import { useGuardianTelemetry } from '../lib/guardianData';

function ModelLoader() {
  return (
    <Html center>
      <div className="flex flex-col items-center gap-2 font-mono text-[9px] text-slate-400 bg-slate-950/80 px-4 py-2.5 rounded-lg border border-blue-500/20 backdrop-blur-md shadow-2xl">
        <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
        <div className="tracking-widest uppercase font-bold text-blue-400 animate-pulse">LOADING DIGITAL TWIN...</div>
      </div>
    </Html>
  );
}

// Spinning particle flow representing dynamic spiraling ventilation airflow (CFD style)
function FanParticles({ active }: { active: boolean }) {
  const count = 120; // Increased particle count for dense fluid flow visualization
  const pointsRef = useRef<THREE.Points>(null);
  
  const [positions, speeds, angles, radii] = React.useMemo(() => {
    const pos = new Float32Array(count * 3);
    const spd = new Float32Array(count);
    const angs = new Float32Array(count);
    const rads = new Float32Array(count);
    
    for (let i = 0; i < count; i++) {
      angs[i] = Math.random() * Math.PI * 2;       // Randomized spiral starting angle
      rads[i] = 0.04 + Math.random() * 0.16;       // Super tight, concentrated baseline flow radius
      spd[i] = 0.6 + Math.random() * 0.9;          // Variable rising speeds
      
      const y = Math.random() * 2.0;
      pos[i * 3 + 1] = y;
      
      // Venturi Tube constriction: flow narrows as it rises
      const currentRad = rads[i] * (1.1 - y * 0.3);
      pos[i * 3] = Math.cos(angs[i]) * currentRad;
      pos[i * 3 + 2] = Math.sin(angs[i]) * currentRad;
    }
    return [pos, spd, angs, rads];
  }, []);

  useFrame((state, delta) => {
    if (!pointsRef.current) return;
    const geo = pointsRef.current.geometry;
    const posAttr = geo.attributes.position;
    if (!posAttr) return;

    // High-fidelity shimmering particle convection diameter
    const material = pointsRef.current.material;
    if (material instanceof THREE.PointsMaterial) {
      material.size = 0.024 + Math.sin(state.clock.elapsedTime * 9) * 0.006;
    }

    const time = state.clock.elapsedTime * 2.4;

    for (let i = 0; i < count; i++) {
      let y = posAttr.getY(i);
      if (active) {
        y += speeds[i] * delta * 1.6;
        if (y > 2.0) {
          y = 0; // Loop seamlessly from floor grid
        }
      } else {
        if (y > 0.01) y -= delta * 0.6; // Soft gravity settle when fan is deactivated
      }

      // Mathematical spiral vortex formulation: angles rotate as they rise over time
      const currentAngle = angles[i] + y * 4.5 + time * (speeds[i] * 0.8);
      // Venturi throat narrowing effect
      const currentRad = radii[i] * (1.15 - y * 0.38);

      posAttr.setX(i, Math.cos(currentAngle) * currentRad);
      posAttr.setY(i, y);
      posAttr.setZ(i, Math.sin(currentAngle) * currentRad);
    }
    posAttr.needsUpdate = true;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        color="#38bdf8"
        size={0.03}
        transparent
        opacity={active ? 0.85 : 0.0}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

// Warning halo under the digital twin model showing real-time risk level
function StatusRing({ risk, isAnomaly }: { risk: string; isAnomaly: boolean }) {
  const ringRef = useRef<THREE.Mesh>(null);
  
  const ringColor = React.useMemo(() => {
    if (isAnomaly || risk === 'Critical') return '#ef4444'; // Red
    if (risk === 'Warning') return '#f59e0b'; // Amber
    return '#10b981'; // Emerald
  }, [risk, isAnomaly]);

  useFrame((state) => {
    if (ringRef.current) {
      const speed = risk === 'Critical' ? 10 : risk === 'Warning' ? 5 : 2;
      const baseScale = 0.98;
      const scale = baseScale + Math.sin(state.clock.elapsedTime * speed) * 0.02;
      ringRef.current.scale.set(scale, scale, 1);
    }
  });

  return (
    <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.15, 0]} receiveShadow>
      <ringGeometry args={[0.3, 0.33, 64]} />
      <meshStandardMaterial 
        color={ringColor} 
        emissive={ringColor} 
        emissiveIntensity={2.8} 
        roughness={0.1}
        metalness={0.9}
        transparent 
        opacity={0.8} 
        side={THREE.DoubleSide} 
      />
    </mesh>
  );
}

// Hazard strobe lighting projecting warning beacons in emergency state
function EmergencyStrobe({ active }: { active: boolean }) {
  const strobeRef = useRef<THREE.PointLight>(null);
  
  useFrame((state) => {
    if (!strobeRef.current) return;
    if (active) {
      strobeRef.current.intensity = Math.sin(state.clock.elapsedTime * 20) > 0 ? 4.0 : 0.05;
    } else {
      strobeRef.current.intensity = 0;
    }
  });

  return <pointLight ref={strobeRef} color="#ef4444" position={[0, 2.0, 0]} distance={5} decay={1.5} />;
}

// Hardware RGB Status LED spotlight projection matching custom client settings
function RGBProjector({ color }: { color: string }) {
  const rgbColor = React.useMemo(() => {
    const c = color ? color.toLowerCase() : '';
    if (c.includes('red')) return '#ef4444';
    if (c.includes('blue')) return '#3b82f6';
    if (c.includes('green')) return '#10b981';
    if (c.includes('yellow')) return '#eab308';
    if (c.includes('cyan')) return '#06b6d4';
    if (c.includes('white')) return '#ffffff';
    return null;
  }, [color]);

  if (!rgbColor) return null;

  return (
    <spotLight
      position={[0, 3.2, 0]}
      angle={0.4}
      penumbra={1}
      intensity={2.8}
      color={rgbColor}
      castShadow
    />
  );
}

function DigitalTwin() {
  const { scene } = useGLTF('/Adaptive Lab Gardian.glb');
  const groupRef = useRef<THREE.Group>(null);
  
  // Sync real-time environmental telemetry state and active manual actuator commands
  const telemetry = useGuardianTelemetry();
  const { meta, action } = telemetry.state;

  // Enable shadows on all meshes of the loaded digital twin model
  React.useEffect(() => {
    scene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
        
        // Calibrate standard materials for premium look
        const mesh = child as THREE.Mesh;
        if (mesh.material && mesh.material instanceof THREE.MeshStandardMaterial) {
          mesh.material.metalness = Math.max(mesh.material.metalness, 0.5);
          mesh.material.roughness = Math.min(mesh.material.roughness, 0.3);
        }
      }
    });
  }, [scene]);

  // Frame loop for dynamic, high-fidelity mesh effects (spinning fan blades, glowing status LEDs)
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.04;
    }

    const speed = meta.risk === 'Critical' ? 12 : meta.risk === 'Warning' ? 5 : 2;
    const pulse = 0.5 + Math.sin(state.clock.elapsedTime * speed) * 0.4;
    
    // Status warning color highlights
    const alertColor = meta.risk === 'Critical' 
      ? new THREE.Color('#f43f5e') 
      : meta.risk === 'Warning' 
        ? new THREE.Color('#fbbf24') 
        : new THREE.Color('#34d399');

    // Hardware indicator spotlight color highlight
    const rgbColorStr = action.rgb_led ? action.rgb_led.toLowerCase() : '';
    const rgbColor = rgbColorStr.includes('red') 
      ? new THREE.Color('#ef4444') 
      : rgbColorStr.includes('blue') 
        ? new THREE.Color('#3b82f6') 
        : rgbColorStr.includes('green') 
          ? new THREE.Color('#10b981') 
          : rgbColorStr.includes('yellow') 
            ? new THREE.Color('#eab308') 
            : rgbColorStr.includes('cyan') 
              ? new THREE.Color('#06b6d4') 
              : new THREE.Color('#38bdf8');

    scene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        if (mesh.material && mesh.material instanceof THREE.MeshStandardMaterial) {
          const name = mesh.name.toLowerCase();

          // 1. Physically spin fan blades if hardware ventilation is active
          if (name.includes('fan') || name.includes('blade') || name.includes('propeller') || name.includes('vent')) {
            if (action.fan === 'ON' || action.fan === '1') {
              mesh.rotation.y += 0.22;
            }
          }
          
          // 2. Project glowing status color onto model LED meshes reactively
          if (name.includes('led') || name.includes('indicator') || name.includes('status') || name.includes('signal')) {
            mesh.material.emissive = alertColor;
            mesh.material.emissiveIntensity = pulse * 3.5;
          }
          
          // 3. Project hardware RGB status colors onto model spotlights or indicator dots
          if (name.includes('rgb') || name.includes('lightdot') || name.includes('spot')) {
            mesh.material.emissive = rgbColor;
            mesh.material.emissiveIntensity = 2.5;
          }
        }
      }
    });
  });

  return (
    <group ref={groupRef}>
      <primitive object={scene} />
    </group>
  );
}

// Fallback visual model in case loading fails or takes too long
function AbstractEquipment({ position, color }: { position: [number, number, number]; color: string }) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime + position[0]) * 0.05;
    }
  });

  return (
    <group position={position}>
      <mesh ref={meshRef} castShadow>
        <boxGeometry args={[1, 1.2, 1]} />
        <meshStandardMaterial color={color} metalness={0.8} roughness={0.2} transparent opacity={0.9} />
      </mesh>
      <gridHelper args={[1.2, 4, color, color]} rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0]} />
    </group>
  );
}

export default function Lab3DModel() {
  const controlsRef = useRef<any>(null);
  const [autoRotate, setAutoRotate] = useState(true);
  const [loadError, setLoadError] = useState(false);

  // Sync real-time environmental telemetry state and active manual actuator commands
  const telemetry = useGuardianTelemetry();
  const { meta, action } = telemetry.state;

  const isAlarmActive = 
    action.alarm === 'ON' || 
    action.alarm === '1' || 
    action.buzzer === 'ON' || 
    action.buzzer === '1';

  const isFanActive = 
    action.fan === 'ON' || 
    action.fan === '1';

  const handleZoomIn = () => {
    if (controlsRef.current) {
      const controls = controlsRef.current;
      const camera = controls.object;
      const target = controls.target;
      const direction = new THREE.Vector3().subVectors(target, camera.position).normalize();
      
      const distance = camera.position.distanceTo(target);
      if (distance > 1.5) {
        camera.position.addScaledVector(direction, 1.0);
        controls.update();
      }
    }
  };

  const handleZoomOut = () => {
    if (controlsRef.current) {
      const controls = controlsRef.current;
      const camera = controls.object;
      const target = controls.target;
      const direction = new THREE.Vector3().subVectors(target, camera.position).normalize();
      
      camera.position.addScaledVector(direction, -1.0);
      controls.update();
    }
  };

  const handleReset = () => {
    if (controlsRef.current) {
      const controls = controlsRef.current;
      const camera = controls.object;
      
      // Smoothly reset camera position and target
      camera.position.set(1.35, 1.35, 1.35);
      controls.target.set(0, 0, 0);
      controls.update();
    }
  };

  return (
    <div className="w-full h-full relative cursor-move bg-slate-950/5">
      <Canvas shadows>
        <PerspectiveCamera makeDefault position={[1.35, 1.35, 1.35]} fov={50} />
        <OrbitControls 
          ref={controlsRef}
          makeDefault 
          enableDamping 
          dampingFactor={0.05} 
          autoRotate={autoRotate} 
          autoRotateSpeed={0.5} 
        />
        
        {/* Responsive Ambient Lighting responding dynamically to threat levels */}
        <ambientLight 
          intensity={meta.risk === 'Critical' ? 0.25 : meta.risk === 'Warning' ? 0.45 : 0.65} 
          color={meta.risk === 'Critical' ? '#fee2e2' : '#ffffff'} 
        />
        <pointLight 
          position={[10, 10, 10]} 
          intensity={meta.risk === 'Critical' ? 0.6 : 1.5} 
          castShadow 
        />
        
        {/* Dynamic emergency visual actors */}
        <EmergencyStrobe active={isAlarmActive} />
        <RGBProjector color={action.rgb_led} />
        <FanParticles active={isFanActive} />
        <StatusRing risk={meta.risk} isAnomaly={meta.is_anomaly} />

        {/* Lab Floor Grid */}
        <Grid 
          infiniteGrid 
          fadeDistance={20} 
          fadeStrength={5} 
          sectionSize={1.5} 
          sectionColor={meta.risk === 'Critical' ? '#be123c' : meta.risk === 'Warning' ? '#d97706' : '#64748b'} 
          cellColor={meta.risk === 'Critical' ? '#ffe4e6' : meta.risk === 'Warning' ? '#fef3c7' : '#cbd5e1'} 
          cellSize={0.5}
        />

        <Suspense fallback={<ModelLoader />}>
          <Center>
            <DigitalTwin />
          </Center>
        </Suspense>

        {loadError && (
          <group position={[0, 0.6, 0]}>
            <AbstractEquipment position={[-2, 0.6, -1]} color="#3b82f6" />
            <AbstractEquipment position={[1.5, 0.6, 0.5]} color="#10b981" />
            <AbstractEquipment position={[-0.5, 0.6, 2]} color="#f43f5e" />
          </group>
        )}

        <Environment preset="city" />
      </Canvas>

      {/* Interactive Controls Overlay (Zoom, Rotate, Reset) */}
      <div className="absolute bottom-4 right-4 flex flex-col gap-2 z-20">
        <button 
          onClick={handleZoomIn}
          title="Zoom In"
          className="w-8 h-8 bg-white/90 hover:bg-white text-slate-600 hover:text-slate-800 border border-slate-200/80 rounded-lg shadow-sm flex items-center justify-center transition-all duration-200 active:scale-95"
        >
          <Plus className="w-4 h-4" />
        </button>
        <button 
          onClick={handleZoomOut}
          title="Zoom Out"
          className="w-8 h-8 bg-white/90 hover:bg-white text-slate-600 hover:text-slate-800 border border-slate-200/80 rounded-lg shadow-sm flex items-center justify-center transition-all duration-200 active:scale-95"
        >
          <Minus className="w-4 h-4" />
        </button>
        <button 
          onClick={handleReset}
          title="Reset Camera Target"
          className="w-8 h-8 bg-white/90 hover:bg-white text-slate-600 hover:text-slate-800 border border-slate-200/80 rounded-lg shadow-sm flex items-center justify-center transition-all duration-200 active:scale-95"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
        <button 
          onClick={() => setAutoRotate(!autoRotate)}
          title={autoRotate ? "Pause Auto-Rotation" : "Start Auto-Rotation"}
          className={`w-8 h-8 border rounded-lg shadow-sm flex items-center justify-center transition-all duration-200 active:scale-95 ${
            autoRotate 
              ? 'bg-blue-50 border-blue-200 text-blue-600 hover:bg-blue-100' 
              : 'bg-white/90 border-slate-200/80 text-slate-600 hover:bg-white hover:text-slate-800'
          }`}
        >
          {autoRotate ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
        </button>
      </div>

      {/* Crosshair Overlay */}
      <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
        <div className="w-12 h-12 border border-blue-500/25 rounded-full flex items-center justify-center">
          <div className="w-1 h-1 bg-blue-500/40 rounded-full shadow-[0_0_10px_rgba(59,130,246,0.4)]" />
        </div>
        <div className="absolute top-1/2 left-4 right-4 h-[1px] bg-blue-500/10" />
        <div className="absolute left-1/2 top-4 bottom-4 w-[1px] bg-blue-500/10" />
      </div>

      {/* Technical Labels Overlay */}
      <div className="absolute bottom-6 left-6 text-[10px] font-tech text-slate-400 space-y-1 select-none pointer-events-none">
        <div className="flex items-center gap-1.5">
          <Compass className="w-3.5 h-3.5 text-blue-500/70" />
          <span>DIGITAL TWIN REALTIME PROJECTION</span>
        </div>
      </div>
    </div>
  );
}

useGLTF.preload('/Adaptive Lab Gardian.glb');
