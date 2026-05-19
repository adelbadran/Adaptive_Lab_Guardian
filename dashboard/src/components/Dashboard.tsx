import React, { useMemo, useState } from 'react';
import {
  Activity,
  Bell,
  Circle,
  Droplets,
  Lock,
  Shield,
  Thermometer,
  Wind,
  Zap,
  Sun,
  Eye,
  Cpu,
  Sliders,
  Volume2,
  VolumeX,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import Lab3DModel from './Lab3DModel';
import { cn } from '../lib/utils';
import { HistoryPoint, useGuardianTelemetry } from '../lib/guardianData';

type Tab = 'guardian' | 'manual' | 'metrics';
type SensorKey = 'Temp_C' | 'Humidity_pct' | 'Gas_AQI' | 'Light_Lux' | 'Motion_Detected';
type Tone = 'blue' | 'emerald' | 'amber' | 'rose' | 'slate';

interface SensorData {
  time: string | number;
  value: number;
}

const toneMap: Record<Tone, { spark: string; bg: string; text: string }> = {
  blue: { spark: '#3b82f6', bg: 'bg-blue-500/10', text: 'text-blue-600' },
  emerald: { spark: '#10b981', bg: 'bg-emerald-500/10', text: 'text-emerald-600' },
  amber: { spark: '#f59e0b', bg: 'bg-amber-500/10', text: 'text-amber-600' },
  rose: { spark: '#f43f5e', bg: 'bg-rose-500/10', text: 'text-rose-600' },
  slate: { spark: '#64748b', bg: 'bg-slate-500/10', text: 'text-slate-600' },
};

const SparkLine = ({ data, color }: { data: SensorData[]; color: string }) => (
  <div className="h-12 w-24">
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data}>
        <Line
          type="monotone"
          dataKey="value"
          stroke={color}
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  </div>
);

const SensorCard = ({
  label,
  value,
  unit,
  icon: Icon,
  tone,
  data,
}: {
  label: string;
  value: string | number;
  unit: string;
  icon: LucideIcon;
  tone: Tone;
  data: SensorData[];
}) => {
  const colors = toneMap[tone];
  return (
    <div className="glass p-4 flex items-center justify-between hover:border-slate-300 transition-all duration-300 group">
      <div className="flex flex-col">
        <span className="text-[10px] uppercase tracking-wider text-slate-400 font-bold mb-1">{label}</span>
        <div className="flex items-baseline gap-1">
          <span className="text-2xl font-tech text-slate-800 leading-none">{value}</span>
          <span className="text-slate-400 text-[10px] font-bold uppercase">{unit}</span>
        </div>
      </div>
      <div className="flex flex-col items-end gap-2">
        <SparkLine data={data} color={colors.spark} />
        <div className={cn('p-1.5 rounded-md', colors.bg)}>
          <Icon className={cn('w-3.5 h-3.5', colors.text)} />
        </div>
      </div>
    </div>
  );
};

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<Tab>('guardian');
  const [pendingMode, setPendingMode] = useState<number | null>(null);
  const [manualNotice, setManualNotice] = useState<string | null>(null);
  const { state, apiReachable, lastError, publishManualMode, publishSystemMode } = useGuardianTelemetry();

  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const prevRiskRef = React.useRef<string | null>(null);

  const speakAlert = React.useCallback((text: string) => {
    if (typeof window === 'undefined' || !window.speechSynthesis) return;
    
    // Stop any ongoing announcement
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    
    // Rigorous and robust English female voice filter that active-filters male voice indicators
    const femaleVoice = voices.find(voice => {
      const name = voice.name.toLowerCase();
      const lang = voice.lang.toLowerCase();
      if (!lang.startsWith('en')) return false;
      // Exclude known male voice indicators to protect female acoustic output
      if (name.includes('male') || name.includes('david') || name.includes('mark') || name.includes('george') || name.includes('ravi') || name.includes('he-')) return false;
      return (
        name.includes('female') || 
        name.includes('zira') || 
        name.includes('samantha') || 
        name.includes('google us english') || 
        name.includes('hazel') ||
        name.includes('susan') ||
        name.includes('cortana') ||
        name.includes('victoria') ||
        name.includes('karen') ||
        name.includes('moira') ||
        name.includes('tessa') ||
        name.includes('veena') ||
        name.includes('fiona') ||
        name.includes('natural') ||
        name.includes('microsoft')
      );
    }) || voices.find(voice => {
      // English voice filter that excludes David/Mark/Male fallbacks
      const name = voice.name.toLowerCase();
      return voice.lang.toLowerCase().startsWith('en') && !name.includes('david') && !name.includes('mark') && !name.includes('male');
    }) || voices.find(voice => voice.lang.toLowerCase().startsWith('en')) || voices[0];

    if (femaleVoice) {
      utterance.voice = femaleVoice;
    }
    
    utterance.pitch = 1.15; // Clean, high-tech synthetic tone
    utterance.rate = 0.95;  // Slightly measured for high-tech authority
    
    window.speechSynthesis.speak(utterance);
  }, []);

  React.useEffect(() => {
    if (!voiceEnabled) return;
    const currentStatus = {
      risk: state.meta.risk,
      isAnomaly: state.meta.is_anomaly,
      label: state.meta.state_label,
      scenario: state.meta.scenario || 'Normal',
      clusterId: state.meta.cluster_id ?? 0,
    };
    
    const statusKey = `${currentStatus.risk}-${currentStatus.isAnomaly}-${currentStatus.label}-${currentStatus.scenario}-${currentStatus.clusterId}`;
    
    if (statusKey !== prevRiskRef.current) {
      const scenarioText = currentStatus.scenario && currentStatus.scenario !== 'Normal' && currentStatus.scenario !== 'Pure IoT Mode'
        ? `Scenario classified as: ${currentStatus.scenario}.`
        : '';
      
      if (currentStatus.isAnomaly || currentStatus.risk === 'Critical' || String(currentStatus.label).toLowerCase() === 'dangerous') {
        speakAlert(`Alert! Critical hazard detected. ${scenarioText} System is in dangerous anomaly state. Initiating automatic cooling and alarms immediately.`);
      } else if (currentStatus.risk === 'Warning' || String(currentStatus.label).toLowerCase() === 'warning') {
        speakAlert(`Warning. Elevated risk detected. ${scenarioText} Engaging mitigation directives.`);
      } else if (currentStatus.risk === 'Nominal' && prevRiskRef.current && !prevRiskRef.current.startsWith('Nominal')) {
        speakAlert("Environmental parameters stabilized. Guardian loop returned to nominal flow.");
      } else if (prevRiskRef.current && currentStatus.scenario !== 'Normal' && currentStatus.scenario !== 'Pure IoT Mode' && !prevRiskRef.current.includes(currentStatus.scenario)) {
        speakAlert(`System transitioned to active cluster: ${currentStatus.scenario}.`);
      }
      prevRiskRef.current = statusKey;
    }
  }, [state.meta.risk, state.meta.is_anomaly, state.meta.state_label, state.meta.scenario, state.meta.cluster_id, voiceEnabled, speakAlert]);

  const toggleVoice = () => {
    const nextState = !voiceEnabled;
    setVoiceEnabled(nextState);
    if (nextState) {
      setTimeout(() => {
        speakAlert("Voice assistant calibrated. Guardian protocol active.");
      }, 100);
    }
  };

  const sensor = state.sensor;
  const meta = state.meta;
  const action = state.action;
  const metrics = state.metrics;
  const history = state.history;
  const isManual = state.system_mode === 'MANUAL';
  const currentTime = new Date();
  const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'LOCAL';

  const tempData = useMemo(() => toSensorData(history, 'Temp_C', sensor.Temp_C), [history, sensor.Temp_C]);
  const humData = useMemo(() => toSensorData(history, 'Humidity_pct', sensor.Humidity_pct), [history, sensor.Humidity_pct]);
  const gasData = useMemo(() => toSensorData(history, 'Gas_AQI', sensor.Gas_AQI), [history, sensor.Gas_AQI]);
  const lightData = useMemo(() => toSensorData(history, 'Light_Lux', sensor.Light_Lux), [history, sensor.Light_Lux]);
  const motionData = useMemo(() => toSensorData(history, 'Motion_Detected', sensor.Motion_Detected), [history, sensor.Motion_Detected]);

  const confidence = clampPercent(meta.guard_confidence <= 1 ? meta.guard_confidence * 100 : meta.guard_confidence);
  const stability = clampPercent(100 - meta.risk_score);
  const throughput = clampPercent((history.length / 80) * 100);
  const stateTone = getStateTone(meta.state_label);
  const liveLabel = state.mqtt.connected ? 'MQTT LIVE' : apiReachable ? 'CSV SYNC' : 'API OFFLINE';
  const lastUpdated = formatTimestamp(state.lastUpdated || sensor.Timestamp);

  const logoRiskColor = useMemo(() => {
    const risk = String(meta.risk || '').toLowerCase();
    if (risk === 'critical' || meta.is_anomaly) return '#ef4444'; // Red alarm
    if (risk === 'warning') return '#f59e0b'; // Amber warning
    return '#10b981'; // Emerald nominal
  }, [meta.risk, meta.is_anomaly]);

  const handleManualMode = async (mode: number) => {
    setPendingMode(mode);
    setManualNotice(null);
    try {
      await publishManualMode(mode);
      setManualNotice(`Manual mode enabled; override sent to ${state.mqtt.actionTopic}: mode ${mode}`);
    } catch (error) {
      setManualNotice(error instanceof Error ? error.message : 'Manual override failed');
    } finally {
      setPendingMode(null);
    }
  };

  return (
    <div className="min-h-screen bg-[#FBFBFC] flex flex-col font-sans text-slate-800">
      <nav className="h-16 px-8 flex items-center justify-between border-b border-slate-100 bg-white z-50 sticky top-0">
        <div className="flex items-center gap-3">
          <div className="relative w-9 h-9 flex items-center justify-center bg-white border border-stone-200/80 rounded-xl shadow-sm group overflow-hidden">
            <div className="absolute inset-0 bg-stone-100/50 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <svg className="w-5.5 h-5.5 text-stone-700 group-hover:scale-110 transition-transform duration-300" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              {/* 1. Shield contour matching the safety status ring boundary, colored in premium rust/bronze-brown */}
              <path d="M12 2L3 7V12C3 16.5 6.5 20.2 12 22C17.5 20.2 21 16.5 21 12V7L12 2Z" stroke="#7c2d12" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="transition-colors duration-500" />
              
              {/* 2. Rotating turbine fan blades in warm rust/bronze-brown (#7c2d12) */}
              <g 
                style={{ 
                  transformOrigin: '12px 12px',
                  animation: (state.action.fan === 'ON' || state.action.fan === '1') ? 'spin 1.5s linear infinite' : 'none'
                }}
              >
                {/* Turbine Blade 1 */}
                <path d="M12 12C12 9.5 13.5 8 15 8C16.5 8 16 10 14.5 11.5C13.5 12.5 12 12 12 12Z" fill="#7c2d12" opacity={0.9} />
                {/* Turbine Blade 2 */}
                <path d="M12 12C9.5 12 8 13.5 8 15C8 16.5 10 16 11.5 14.5C12.5 13.5 12 12 12 12Z" fill="#7c2d12" opacity={0.9} />
                {/* Turbine Blade 3 */}
                <path d="M12 12C12 14.5 10.5 16 9 16C7.5 16 8 14 9.5 12.5C10.5 11.5 12 12 12 12Z" fill="#7c2d12" opacity={0.9} />
                {/* Turbine Blade 4 */}
                <path d="M12 12C14.5 12 16 10.5 16 9C16 7.5 14 8 12.5 9.5C11.5 10.5 12 12 12 12Z" fill="#7c2d12" opacity={0.9} />
              </g>

              {/* 3. Center indicator status node matching 3D model status LED colors */}
              <circle cx="12" cy="12" r="2.2" fill={logoRiskColor} className="animate-pulse transition-colors duration-500" />
            </svg>
          </div>
          <span className="tech-font text-xs tracking-widest uppercase font-bold">Adaptive Lab Guardian</span>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-4 text-slate-400">
            {/* High-End Voice Assistant Toggler */}
            <button
              onClick={toggleVoice}
              title={voiceEnabled ? "Mute Voice Assistant (Active)" : "Unmute Voice Assistant (Muted)"}
              className={cn(
                "p-1.5 px-2.5 rounded-lg border transition-all duration-300 flex items-center gap-1.5 active:scale-95",
                voiceEnabled
                  ? "bg-blue-50 border-blue-200 text-blue-600 shadow-sm"
                  : "bg-slate-50 border-slate-200 text-slate-400 hover:text-slate-500"
              )}
            >
              {voiceEnabled ? (
                <>
                  <Volume2 className="w-[15px] h-[15px] animate-pulse" strokeWidth={2.5} />
                  <span className="text-[9px] font-bold uppercase tracking-wider hidden sm:inline">VOICE ON</span>
                </>
              ) : (
                <>
                  <VolumeX className="w-[15px] h-[15px]" strokeWidth={2.5} />
                  <span className="text-[9px] font-bold uppercase tracking-wider hidden sm:inline">MUTED</span>
                </>
              )}
            </button>

            <div className="hidden lg:block text-right">
              <p className="text-[10px] font-bold text-slate-600 leading-none">
                {currentTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })}
              </p>
              <p className="text-[8px] uppercase tracking-widest mt-0.5">{timeZone}</p>
            </div>

            <div className="relative">
              <button className="p-2 text-slate-400 hover:text-slate-600 transition-colors" title={state.mqtt.error || liveLabel}>
                <Bell className="w-[18px] h-[18px]" strokeWidth={2.5} />
                <span className={cn('absolute top-1.5 right-1.5 w-2 h-2 rounded-full border-2 border-white', state.mqtt.connected ? 'bg-emerald-500' : 'bg-amber-500')} />
              </button>
            </div>
          </div>
        </div>
      </nav>

      <main className="flex-1 flex flex-col gap-4 p-4 lg:p-6 max-w-[1600px] mx-auto w-full overflow-hidden">
        <div className="grid grid-cols-1 lg:grid-cols-10 gap-4 min-h-[550px]">
          <section className="lg:col-span-6 glass overflow-hidden relative flex flex-col">
            <div className="absolute inset-0 grid-pattern opacity-40 pointer-events-none"></div>
            <div className="absolute top-4 left-4 z-10">
              <p className="text-[10px] font-tech text-slate-400 uppercase tracking-widest">
                Lab Digital Twin
              </p>
            </div>

            {/* Floating Dynamic Island System Status HUD (White & Brown Palette) */}
            <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20 pointer-events-auto">
              <motion.div 
                layout
                className={cn(
                  "flex items-center gap-2 px-3 py-1 rounded-full border shadow-lg backdrop-blur-xl transition-all duration-500 ease-out",
                  meta.risk === 'Critical' 
                    ? "bg-amber-950/95 border-amber-600/40 text-amber-100 min-w-[210px]" 
                    : meta.risk === 'Warning' 
                      ? "bg-amber-50/95 border-amber-200 text-amber-900 min-w-[170px]" 
                      : "bg-white/95 border-stone-200/90 text-stone-800 min-w-[130px]"
                )}
                style={
                  meta.risk === 'Critical'
                    ? (() => {
                        const scen = String(meta.scenario || '').toLowerCase();
                        const clusterId = meta.cluster_id;
                        if (scen.includes('security') || scen.includes('breach') || clusterId === 3) {
                          return { backgroundColor: '#1a0b0e', borderColor: '#e11d48' }; // Crimson red border/bg for Security
                        }
                        return { backgroundColor: '#1c0d02', borderColor: '#7c2d12' }; // Deep orange/brown border/bg for Chemical/Thermal
                      })()
                    : {}
                }
              >
                {/* Status indicator node */}
                <div className="relative flex items-center justify-center w-2.5 h-2.5">
                  <span className={cn(
                    "absolute w-2 h-2 rounded-full border border-white/20 animate-ping",
                    meta.risk === 'Critical' ? "bg-rose-400" : meta.risk === 'Warning' ? "bg-amber-400" : "bg-emerald-400"
                  )} />
                  <span className={cn(
                    "relative w-1.5 h-1.5 rounded-full",
                    meta.risk === 'Critical' ? "bg-rose-500" : meta.risk === 'Warning' ? "bg-amber-500" : "bg-emerald-500"
                  )} />
                </div>

                {/* Status Text content */}
                <div className="flex flex-col flex-1 overflow-hidden select-none">
                  {meta.risk === 'Critical' ? (
                    <motion.div 
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="flex items-center justify-between gap-2.5 text-left w-full"
                    >
                      <div>
                        {(() => {
                          const scen = String(meta.scenario || '').toLowerCase();
                          const clusterId = meta.cluster_id;
                          if (scen.includes('security') || scen.includes('breach') || clusterId === 3) {
                            return (
                              <>
                                <p className="text-[7px] uppercase tracking-widest text-rose-400 font-bold leading-none">SECURITY BREACH</p>
                                <p className="text-[9px] font-tech text-rose-100 font-bold leading-tight mt-0.5">INTRUSION DETECTED</p>
                              </>
                            );
                          }
                          if (scen.includes('chemical') || scen.includes('gas') || clusterId === 2) {
                            return (
                              <>
                                <p className="text-[7px] uppercase tracking-widest text-amber-400 font-bold leading-none">CHEMICAL HAZARD</p>
                                <p className="text-[9px] font-tech text-amber-100 font-bold leading-tight mt-0.5">GAS LEAK DETECTED</p>
                              </>
                            );
                          }
                          if (scen.includes('crowded') || scen.includes('temp') || clusterId === 1) {
                            return (
                              <>
                                <p className="text-[7px] uppercase tracking-widest text-amber-400 font-bold leading-none">THERMAL HAZARD</p>
                                <p className="text-[9px] font-tech text-amber-100 font-bold leading-tight mt-0.5">OVERHEAT DETECTED</p>
                              </>
                            );
                          }
                          return (
                            <>
                              <p className="text-[7px] uppercase tracking-widest text-amber-400 font-bold leading-none">HAZARD CRITICAL</p>
                              <p className="text-[9px] font-tech text-amber-100 font-bold leading-tight mt-0.5">ANOMALY DETECTED</p>
                            </>
                          );
                        })()}
                      </div>
                    </motion.div>
                  ) : meta.risk === 'Warning' ? (
                    <motion.div 
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="flex items-center justify-between gap-2 text-left w-full"
                    >
                      <div>
                        <p className="text-[7px] uppercase tracking-widest text-amber-800 font-bold leading-none">SYSTEM WARNING</p>
                        <p className="text-[9px] font-tech text-amber-700 font-bold leading-tight mt-0.5">RISK ELEVATED</p>
                      </div>
                      <span className="text-[7.5px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-800 font-bold shrink-0 leading-none">
                        {isManual ? 'AI PAUSED' : 'AI ENGAGED'}
                      </span>
                    </motion.div>
                  ) : (
                    <motion.div 
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="text-left leading-tight py-0.5"
                    >
                      <p className="text-[6.5px] uppercase tracking-widest text-stone-500 font-bold leading-none">SYSTEM SECURE</p>
                      <p className="text-[8.5px] font-tech text-stone-800 font-bold leading-none mt-0.5">NOMINAL STATE</p>
                    </motion.div>
                  )}
                </div>
              </motion.div>
            </div>

            <div className="flex-1 relative">
              <Lab3DModel />
            </div>
          </section>

          <section className="lg:col-span-4 flex flex-col gap-3">
            <div className="flex items-center justify-between px-2">
              <h3 className="tech-font text-[10px] uppercase tracking-widest">Live Stream</h3>
              <span className={cn('flex items-center gap-1.5 text-[10px] font-bold px-2 py-0.5 rounded border', stateTone.badge)}>
                <Circle className="w-2 h-2 fill-current" />
                {liveLabel}
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <SensorCard
                label="Temperature"
                value={formatNumber(sensor.Temp_C, 1)}
                unit="C"
                icon={Thermometer}
                tone="blue"
                data={tempData}
              />
              <SensorCard
                label="Humidity"
                value={formatNumber(sensor.Humidity_pct, 1)}
                unit="%"
                icon={Droplets}
                tone="emerald"
                data={humData}
              />
              <SensorCard
                label="Gas AQI"
                value={formatNumber(sensor.Gas_AQI, 0)}
                unit="AQI"
                icon={Wind}
                tone={meta.state_label === 'Dangerous' ? 'rose' : 'amber'}
                data={gasData}
              />
              <SensorCard
                label="Light (Lux)"
                value={formatNumber(sensor.Light_Lux, 0)}
                unit="Lux"
                icon={Sun}
                tone="blue"
                data={lightData}
              />
              <div className="sm:col-span-2">
                <SensorCard
                  label="Motion"
                  value={sensor.Motion_Detected ? 'YES' : 'NO'}
                  unit="PIR"
                  icon={Eye}
                  tone="emerald"
                  data={motionData}
                />
              </div>
            </div>

            <div className={cn('border-2 border-dashed rounded-xl flex flex-col justify-center min-h-[110px] px-5 py-4 mt-1', stateTone.panel)}>
              <div className="flex items-center justify-between mb-3">
                <h3 className="tech-font text-[10px] uppercase tracking-widest font-bold text-slate-500">Anomaly Engine</h3>
                <span className={cn('flex items-center gap-1.5 text-[8px] font-bold px-2 py-0.5 rounded border tracking-wider', stateTone.badge)}>
                  <Circle className={cn('w-1.5 h-1.5 fill-current', !meta.is_anomaly && 'animate-pulse')} />
                  {isManual ? 'AI MODELS PAUSED' : meta.is_anomaly ? 'VARIANCE DETECTED' : 'SCANNING LIVE PATTERNS'}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-3 text-[10px]">
                <MiniMetric label="Anomaly" value={formatNumber(meta.anomaly_score, 2)} unit="art2" />
                <MiniMetric label="Trend" value={formatNumber(meta.trend, 1)} unit="rbf" />
                <MiniMetric label="Cluster" value={meta.scenario} unit="som" />
              </div>

              {/* Mode Activation Selector (AI / MANUAL) */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mt-4 pt-3 border-t border-slate-100/50">
                <span className="text-[9px] font-semibold text-slate-400 leading-snug">
                  Configure the active operational control loop methodology:
                </span>
                <div className="flex items-center gap-4 sm:ml-auto">
                  {/* AI Radio Button */}
                  <label className="flex items-center gap-2 cursor-pointer select-none">
                    <input
                      type="radio"
                      name="system-mode"
                      checked={state.system_mode === 'AI'}
                      onChange={() => publishSystemMode('AI')}
                      disabled={!apiReachable}
                      className="sr-only"
                    />
                    <div className={cn(
                      "w-3.5 h-3.5 rounded-full border flex items-center justify-center transition-all duration-200",
                      state.system_mode === 'AI'
                        ? "border-blue-500 bg-blue-50/50 shadow-sm"
                        : "border-slate-300 bg-white"
                    )}>
                      {state.system_mode === 'AI' && (
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
                      )}
                    </div>
                    <span className={cn(
                      "text-[9px] font-bold tracking-wider transition-colors duration-200",
                      state.system_mode === 'AI' ? "text-blue-600 font-extrabold" : "text-slate-500"
                    )}>
                      AI AUTOMATIC
                    </span>
                  </label>

                  {/* Manual Radio Button */}
                  <label className="flex items-center gap-2 cursor-pointer select-none">
                    <input
                      type="radio"
                      name="system-mode"
                      checked={state.system_mode === 'MANUAL'}
                      onChange={() => publishSystemMode('MANUAL')}
                      disabled={!apiReachable}
                      className="sr-only"
                    />
                    <div className={cn(
                      "w-3.5 h-3.5 rounded-full border flex items-center justify-center transition-all duration-200",
                      state.system_mode === 'MANUAL'
                        ? "border-amber-500 bg-amber-50/50 shadow-sm"
                        : "border-slate-300 bg-white"
                    )}>
                      {state.system_mode === 'MANUAL' && (
                        <div className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                      )}
                    </div>
                    <span className={cn(
                      "text-[9px] font-bold tracking-wider transition-colors duration-200",
                      state.system_mode === 'MANUAL' ? "text-amber-600 font-extrabold" : "text-slate-500"
                    )}>
                      MANUAL OVERRIDE
                    </span>
                  </label>
                </div>
              </div>
            </div>
          </section>
        </div>

        <section className="glass flex flex-col min-h-[300px] overflow-hidden">
          <div className="flex border-b border-brand-border h-11 bg-white/50">
            <TabHead active={activeTab === 'guardian'} onClick={() => setActiveTab('guardian')}>AI Guardian Mode</TabHead>
            <TabHead active={activeTab === 'manual'} onClick={() => setActiveTab('manual')}>Manual Override</TabHead>
            <TabHead active={activeTab === 'metrics'} onClick={() => setActiveTab('metrics')}>Performance Metrics</TabHead>
          </div>

          <div className="flex-1 p-6 overflow-hidden">
            <AnimatePresence mode="wait">
              {activeTab === 'guardian' && (
                <motion.div
                  key="guardian"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="grid grid-cols-1 lg:grid-cols-3 gap-8 h-full"
                >
                  <div className="lg:col-span-2 flex flex-col gap-4">
                    <div className="flex items-center justify-between">
                      <h4 className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">Automated Log // System State</h4>
                      <span className="text-[9px] text-slate-400 tracking-tighter">Last updated {lastUpdated}</span>
                    </div>
                    <div className="space-y-2.5 overflow-y-auto pr-2">
                      {state.logs.slice(0, 3).map((log) => (
                        <div key={log.id} className={cn(
                          'flex items-start gap-4 p-3 rounded-lg border transition-colors',
                          log.type === 'warning' ? 'bg-amber-50/30 border-amber-100/50' : 'bg-white border-brand-border',
                        )}>
                          <span className="text-[10px] font-mono text-slate-400 mt-0.5">{log.time}</span>
                          <div className="flex-1">
                            <p className="text-[11px] font-semibold text-slate-700 leading-snug">{log.message}</p>
                            <p className="text-[10px] text-slate-500 mt-1">
                              Status: {log.type === 'warning' ? 'INTERVENTION_REQUIRED' : 'LOGGED_NOMINAL'}
                            </p>
                          </div>
                          {log.type === 'success' && <span className="text-[9px] px-2 py-0.5 bg-green-100 text-green-700 rounded-full font-bold">RESOLVED</span>}
                        </div>
                      ))}
                      {lastError && (
                        <div className="flex items-start gap-4 p-3 rounded-lg border bg-amber-50/30 border-amber-100/50">
                          <span className="text-[10px] font-mono text-slate-400 mt-0.5">API</span>
                          <div className="flex-1">
                            <p className="text-[11px] font-semibold text-slate-700 leading-snug">{lastError}</p>
                            <p className="text-[10px] text-slate-500 mt-1">Status: USING_LOCAL_FALLBACK</p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex flex-col gap-6 border-l border-brand-border pl-8">
                    <h4 className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">Model Confidence</h4>
                    <div className="flex items-center gap-6">
                      <div className="relative w-16 h-16">
                        <svg viewBox="0 0 36 36" className="w-full h-full transform -rotate-90">
                          <circle cx="18" cy="18" r="16" fill="none" stroke="#f1f5f9" strokeWidth="4" />
                          <circle cx="18" cy="18" r="16" fill="none" stroke="#3b82f6" strokeWidth="4" strokeDasharray={`${confidence}, 100`} />
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center">
                          <span className="text-[10px] font-bold">{formatNumber(confidence, 0)}%</span>
                        </div>
                      </div>
                      <div className="flex flex-col">
                        <p className="text-[10px] font-semibold uppercase text-slate-400 tracking-wider">Action Mode</p>
                        <p className="font-tech text-lg">{action.action_id}<span className="text-[10px] text-slate-400 ml-1">ESP32</span></p>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <ProgressLine label="CORE STABILITY" value={stability} color="bg-blue-500" />
                      <ProgressLine label="DATA THROUGHPUT" value={throughput} color="bg-slate-900" />
                    </div>
                  </div>
                </motion.div>
              )}

              {activeTab === 'manual' && (
                <motion.div
                  key="manual"
                  className="grid grid-cols-2 md:grid-cols-4 gap-4"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <OverrideToggle label="Normal Reset" icon={Shield} active={action.action_id === 0} disabled={!apiReachable || pendingMode !== null} pending={pendingMode === 0} onClick={() => handleManualMode(0)} />
                  <OverrideToggle label="Ventilation" icon={Wind} active={action.action_id === 1} disabled={!apiReachable || pendingMode !== null} pending={pendingMode === 1} onClick={() => handleManualMode(1)} />
                  <OverrideToggle label="Chemical Alert" icon={Zap} active={action.action_id === 2} disabled={!apiReachable || pendingMode !== null} pending={pendingMode === 2} onClick={() => handleManualMode(2)} />
                  <OverrideToggle label="Security Breach" icon={Lock} active={action.action_id === 3} disabled={!apiReachable || pendingMode !== null} pending={pendingMode === 3} onClick={() => handleManualMode(3)} />
                  {manualNotice && (
                    <div className="col-span-2 md:col-span-4 text-[10px] font-bold uppercase tracking-widest text-slate-500 bg-slate-50 border border-brand-border rounded-lg px-3 py-2">
                      {manualNotice}
                    </div>
                  )}
                </motion.div>
              )}

              {activeTab === 'metrics' && (
                <motion.div
                  key="metrics"
                  className="flex flex-col lg:flex-row gap-8 items-stretch"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  {/* Left Column: Overall Training Metrics Stack */}
                  <div className="flex flex-col justify-start items-center gap-8 lg:w-1/4 pb-4 lg:pb-0 lg:border-r border-brand-border pr-8">
                    <div className="text-center lg:text-left w-full">
                      <h4 className="text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-1">Model Accuracy</h4>
                      <p className="text-[9px] text-slate-400 leading-snug">Validated model optimization metrics derived from offline training compilation cycles.</p>
                    </div>
                    <div className="flex flex-row lg:flex-col gap-8 items-center justify-around w-full overflow-y-auto max-h-[850px] pr-2">
                      <MetricCircle label="Risk Classifier Accuracy" value={metrics.accuracy} color="emerald" />
                      <MetricCircle label="Topological Scenario Rate" value={metrics.scenarioAccuracy} color="blue" />
                      <MetricCircle label="GNN Spatial Attention Fit" value={metrics.gnnAccuracy} color="blue" />
                      <MetricCircle label="PCA Variance Coverage" value={metrics.pcaExplained} color="slate" />
                      <MetricCircle label="Fuzzy Inference Precision" value={100 - metrics.falseAlertRate} color="amber" />
                      <MetricCircle label="RL DQN Policy Success" value={100 - (metrics.warningMissRate * 8)} color="rose" />
                    </div>
                  </div>

                  {/* Right Column: Dynamic Neural Pipeline Directory */}
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">Neural Pipeline Directory</h4>
                      <span className="text-[8px] font-bold text-blue-600 bg-blue-50 border border-blue-100 px-2 py-0.5 rounded animate-pulse">
                        LIVE RECURRENT SYNC
                      </span>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* GNN Card */}
                      <div className="bg-slate-50/40 border border-brand-border rounded-xl p-4 flex flex-col justify-between hover:border-slate-300 transition-all duration-300">
                        <div>
                          <div className="flex items-center justify-between gap-2 mb-2">
                            <span className="text-[9px] font-tech text-blue-600 bg-blue-50/50 border border-blue-200 px-2 py-0.5 rounded font-bold">GNN</span>
                            <span className="text-[9px] font-bold text-slate-400 font-mono">SPATIAL ESTIMATION</span>
                          </div>
                          
                          <div className="relative w-full h-[120px] my-1">
                            {(() => {
                              const risk = meta.spatial_risk || 0;
                              const intensity = Math.min(1, risk * 1.5);
                              const color = risk > 0.65 ? '#ef4444' : risk > 0.4 ? '#f59e0b' : '#3b82f6';
                              const nodes = [
                                { x: 50, y: 15, label: 'TMP' },
                                { x: 85, y: 40, label: 'HUM' },
                                { x: 72, y: 80, label: 'GAS' },
                                { x: 28, y: 80, label: 'LUX' },
                                { x: 15, y: 40, label: 'MOT' },
                              ];
                              return (
                                <div className="absolute inset-0 flex items-center justify-center">
                                  <svg viewBox="0 0 100 100" className="w-full h-full max-w-[120px] overflow-visible">
                                    {nodes.map((n1, i) => 
                                      nodes.slice(i + 1).map((n2, j) => (
                                        <line
                                          key={`edge-${i}-${j}`}
                                          x1={n1.x} y1={n1.y}
                                          x2={n2.x} y2={n2.y}
                                          stroke={color}
                                          strokeWidth={0.5 + intensity * 2}
                                          opacity={0.15 + intensity * 0.5}
                                          className={intensity > 0.5 ? "animate-pulse" : ""}
                                        />
                                      ))
                                    )}
                                    {nodes.map((n, i) => (
                                      <g key={`node-${i}`}>
                                        <circle cx={n.x} cy={n.y} r={9} fill="white" stroke={color} strokeWidth={1.5} />
                                        <text x={n.x} y={n.y + 2} fontSize="5" fontWeight="bold" fill="#475569" textAnchor="middle">
                                          {n.label}
                                        </text>
                                      </g>
                                    ))}
                                  </svg>
                                </div>
                              );
                            })()}
                          </div>
                        </div>
                        <div className="mt-2 pt-3 border-t border-slate-200/50 flex justify-between text-[8px] font-mono">
                          <span className="text-slate-400">NODES: 5 | EDGES: {metrics.gnnEdges ?? 20}</span>
                          <span className={cn("font-bold", (meta.spatial_risk || 0) > 0.65 ? "text-rose-600" : "text-emerald-600")}>
                            SPATIAL RISK: {formatNumber(meta.spatial_risk, 2)}
                          </span>
                        </div>
                      </div>

                      {/* ART2 Card */}
                      <div className="bg-slate-50/40 border border-brand-border rounded-xl p-4 flex flex-col justify-between hover:border-slate-300 transition-all duration-300">
                        <div>
                          <div className="flex items-center justify-between gap-2 mb-2">
                            <span className="text-[9px] font-tech text-rose-600 bg-rose-50/50 border border-rose-200 px-2 py-0.5 rounded font-bold">ART2</span>
                            <span className="text-[9px] font-bold text-slate-400 font-mono">ANOMALY DETECTOR</span>
                          </div>
                          <p className="text-[11px] font-semibold text-slate-700 mb-1">Adaptive Resonance Theory 2</p>
                          <p className="text-[9px] text-slate-500 leading-relaxed">
                            Performs real-time unsupervised pattern matching using a dual-layer feedback resonance network. Continuously evaluates vigilance criteria against multi-dimensional sensor input vectors to detect statistical anomalies.
                          </p>
                        </div>
                        <div className="mt-4 pt-3 border-t border-slate-200/50 flex justify-between text-[8px] font-mono">
                          <span className="text-slate-400">VIGILANCE: 0.85 | CATEGORIES: {metrics.art2Categories ?? 5}</span>
                          <span className="font-bold text-rose-600">ANOMALY: {formatNumber(meta.anomaly_score, 2)}</span>
                        </div>
                      </div>

                      {/* RBF Card */}
                      <div className="bg-slate-50/40 border border-brand-border rounded-xl p-4 flex flex-col justify-between hover:border-slate-300 transition-all duration-300">
                        <div>
                          <div className="flex items-center justify-between gap-2 mb-2">
                            <span className="text-[9px] font-tech text-amber-600 bg-amber-50/50 border border-amber-200 px-2 py-0.5 rounded font-bold">RBF</span>
                            <span className="text-[9px] font-bold text-slate-400 font-mono">TREND TRAJECTORY</span>
                          </div>
                          <p className="text-[11px] font-semibold text-slate-700 mb-1">Radial Basis Function</p>
                          <p className="text-[9px] text-slate-500 leading-relaxed">
                            Approximates non-linear risk state trajectories via multidimensional Gaussian activation functions. Formulates predictive gradient slope vectors to model environmental stability trends.
                          </p>
                        </div>
                        <div className="mt-4 pt-3 border-t border-slate-200/50 flex justify-between text-[8px] font-mono">
                          <span className="text-slate-400">SIGMA: {metrics.rbfSigma ?? 0.98}</span>
                          <span className="font-bold text-amber-600">TREND: {formatNumber(meta.trend, 1)}</span>
                        </div>
                      </div>

                      {/* SOM Card */}
                      <div className="bg-slate-50/40 border border-brand-border rounded-xl p-4 flex flex-col justify-between hover:border-slate-300 transition-all duration-300">
                        <div>
                          <div className="flex items-center justify-between gap-2 mb-2">
                            <span className="text-[9px] font-tech text-emerald-600 bg-emerald-50/50 border border-emerald-200 px-2 py-0.5 rounded font-bold">SOM</span>
                            <span className="text-[9px] font-bold text-slate-400 font-mono">SCENARIO CLUSTER</span>
                          </div>
                          <p className="text-[11px] font-semibold text-slate-700 mb-1">Self-Organizing Map</p>
                          <p className="text-[9px] text-slate-500 leading-relaxed">
                            Maps high-dimensional topological sensor spaces onto a discretized two-dimensional feature lattice. Classifies operational profiles into localized clusters representing specific safety states.
                          </p>
                        </div>
                        <div className="mt-4 pt-3 border-t border-slate-200/50 flex justify-between text-[8px] font-mono">
                          <span className="text-slate-400">CLUSTER: ID {meta.cluster_id}</span>
                          <span className="font-bold text-emerald-600">SCENARIO: {meta.scenario}</span>
                        </div>
                      </div>

                      {/* FUZZY Card */}
                      <div className="bg-slate-50/40 border border-brand-border rounded-xl p-4 flex flex-col justify-between hover:border-slate-300 transition-all duration-300">
                        <div>
                          <div className="flex items-center justify-between gap-2 mb-2">
                            <span className="text-[9px] font-tech text-blue-600 bg-blue-50/50 border border-blue-200 px-2 py-0.5 rounded font-bold">FUZZY</span>
                            <span className="text-[9px] font-bold text-slate-400 font-mono">RISK DECISION ENGINE</span>
                          </div>
                          <p className="text-[11px] font-semibold text-slate-700 mb-1">Fuzzy Logic Controller</p>
                          <p className="text-[9px] text-slate-500 leading-relaxed">
                            Maps continuous physical environmental states into linguistic membership classifications. Applies standard rules to formulate accurate spatial risk coefficients and safety classifications.
                          </p>
                        </div>
                        <div className="mt-4 pt-3 border-t border-slate-200/50 flex justify-between text-[8px] font-mono">
                          <span className="text-slate-400">RULES: 27 INTERFERENCES</span>
                          <span className="font-bold text-blue-600">RISK: {formatNumber(meta.risk_score, 1)}% ({meta.risk})</span>
                        </div>
                      </div>

                      {/* RL Card */}
                      <div className="bg-slate-50/40 border border-brand-border rounded-xl p-4 flex flex-col justify-between hover:border-slate-300 transition-all duration-300">
                        <div>
                          <div className="flex items-center justify-between gap-2 mb-2">
                            <span className="text-[9px] font-tech text-purple-600 bg-purple-50/50 border border-purple-200 px-2 py-0.5 rounded font-bold">RL</span>
                            <span className="text-[9px] font-bold text-slate-400 font-mono">ACTUATOR OPTIMIZER</span>
                          </div>
                          <p className="text-[11px] font-semibold text-slate-700 mb-1">Reinforcement Learning</p>
                          <p className="text-[9px] text-slate-500 leading-relaxed">
                            Formulates optimal safety policy directives via deep Q-network feedback optimization. Balances power conservation, mechanical structural wear-and-tear, and emergency response activation timing.
                          </p>
                        </div>
                        <div className="mt-4 pt-3 border-t border-slate-200/50 flex justify-between text-[8px] font-mono">
                          <span className="text-slate-400">POLICY: DQN (TARGET NETWORK)</span>
                          <span className="font-bold text-purple-600">ACTION ID: {action.action_id}</span>
                        </div>
                      </div>

                      {/* PCA Card */}
                      <div className="bg-slate-50/40 border border-brand-border rounded-xl p-4 flex flex-col justify-between hover:border-slate-300 transition-all duration-300">
                        <div>
                          <div className="flex items-center justify-between gap-2 mb-2">
                            <span className="text-[9px] font-tech text-slate-600 bg-slate-100 border border-slate-200 px-2 py-0.5 rounded font-bold">PCA</span>
                            <span className="text-[9px] font-bold text-slate-400 font-mono">FEATURE EXTRACTION</span>
                          </div>
                          <p className="text-[11px] font-semibold text-slate-700 mb-1">Principal Component Analysis</p>
                          <p className="text-[9px] text-slate-500 leading-relaxed">
                            Compresses high-dimensional raw physical sensor telemetry into decorrelated orthogonal vectors. Isolates environmental sensor cross-chatter, ambient noise, and dynamic measurement fluctuations.
                          </p>
                        </div>
                        <div className="mt-4 pt-3 border-t border-slate-200/50 flex justify-between text-[8px] font-mono">
                          <span className="text-slate-400">COMPONENTS: 2 PRINCIPAL AXES</span>
                          <span className="font-bold text-slate-600">EXPLAINED: {formatNumber(metrics.pcaExplained, 1)}%</span>
                        </div>
                      </div>

                      {/* GA Card */}
                      <div className="bg-slate-50/40 border border-brand-border rounded-xl p-4 flex flex-col justify-between hover:border-slate-300 transition-all duration-300">
                        <div>
                          <div className="flex items-center justify-between gap-2 mb-2">
                            <span className="text-[9px] font-tech text-pink-600 bg-pink-50/50 border border-pink-200 px-2 py-0.5 rounded font-bold">GA</span>
                            <span className="text-[9px] font-bold text-slate-400 font-mono">THRESHOLD EVOLUTION</span>
                          </div>
                          <p className="text-[11px] font-semibold text-slate-700 mb-1">Genetic Algorithm</p>
                          <p className="text-[9px] text-slate-500 leading-relaxed">
                            Simulates natural selection to dynamically evolve risk boundaries and state thresholds. Mutates decision parameters over generations to perfectly adapt the fuzzy logic boundaries to the specific laboratory environment.
                          </p>
                        </div>
                        <div className="mt-4 pt-3 border-t border-slate-200/50 flex justify-between text-[8px] font-mono">
                          <span className="text-slate-400">FITNESS: {metrics.gaScore ?? 0.4516}</span>
                          <span className="font-bold text-pink-600 animate-pulse">GENERATION: ACTIVE</span>
                        </div>
                      </div>
                    </div>

                    {/* Spatial Network Attention Mapping */}
                    <div className="mt-6 p-4 border border-brand-border rounded-xl bg-slate-50/20">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex flex-col">
                          <h4 className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">Spatial Attention Topology</h4>
                          <p className="text-[9px] text-slate-400">Dynamic GATConv correlation mapping computed across environmental zones.</p>
                        </div>
                        <span className="text-[8px] font-bold text-emerald-600 bg-emerald-50 border border-emerald-100 px-2 py-0.5 rounded animate-pulse">
                          RESOLVED TOPO MAP
                        </span>
                      </div>
                      <div className="flex flex-col md:flex-row gap-6 items-center">
                        <div className="w-full md:w-1/3 flex justify-center bg-white border border-slate-200/60 rounded-xl p-4 shadow-sm">
                          <img
                            src="/gnn_attention_graph.png"
                            alt="GNN Spatial Attention Graph"
                            className="max-h-[180px] object-contain hover:scale-105 transition-transform duration-300"
                          />
                        </div>
                        <div className="flex-1 space-y-3.5">
                          <h5 className="text-[10px] font-bold text-slate-700 uppercase tracking-wide">Graph Convolution Interpretation // Message-Passing:</h5>
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-[9px] leading-relaxed">
                            <div className="bg-white/60 p-3 rounded-lg border border-slate-100">
                              <span className="font-bold text-blue-600 block mb-1">THERMODYNAMIC VECTOR</span>
                              <p className="text-slate-500">
                                Correlates Temperature and Humidity nodes. Detects joint ambient energy signatures that indicate hot-air boundaries.
                              </p>
                            </div>
                            <div className="bg-white/60 p-3 rounded-lg border border-slate-100">
                              <span className="font-bold text-rose-600 block mb-1">ATMOSPHERIC DIFFUSION</span>
                              <p className="text-slate-500">
                                Integrates the Gas sensor node into surrounding thermodynamic neighborhoods to trigger predictive safety warnings.
                              </p>
                            </div>
                            <div className="bg-white/60 p-3 rounded-lg border border-slate-100">
                              <span className="font-bold text-emerald-600 block mb-1">OCCUPANCY VECTOR</span>
                              <p className="text-slate-500">
                                Binds Light and Motion nodes, estimating human occupancy profiles to isolate structural anomalies.
                              </p>
                            </div>
                            <div className="bg-white/60 p-3 rounded-lg border border-slate-100">
                              <span className="font-bold text-slate-600 block mb-1">TEMPORAL RESIDUALS</span>
                              <p className="text-slate-500">
                                Feeds spatial pooling representations into the fully connected classifier layer to isolate persistent anomalies.
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </section>
      </main>
    </div>
  );
}

function TabHead({ children, active, onClick }: { children: React.ReactNode; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'flex-1 px-4 h-full font-tech text-[9px] uppercase tracking-[0.2em] transition-all border-b-2 border-r border-brand-border last:border-r-0',
        active ? 'border-b-slate-900 text-slate-900 font-bold bg-white' : 'border-b-transparent text-slate-400 hover:text-slate-600',
      )}
    >
      {children}
    </button>
  );
}

function ProgressLine({ label, value, color }: { label: string; value: number; color: string }) {
  const safeValue = clampPercent(value);
  return (
    <div>
      <div className="flex justify-between text-[10px] mb-1 font-bold tracking-wider text-slate-500">
        <span>{label}</span>
        <span>{formatNumber(safeValue, 0)}%</span>
      </div>
      <div className="w-full h-1 bg-slate-100 rounded-full overflow-hidden">
        <div className={cn('h-full rounded-full transition-all duration-1000', color)} style={{ width: `${safeValue}%` }} />
      </div>
    </div>
  );
}

function ControlBtn({ children }: { children: React.ReactNode }) {
  return (
    <button className="w-8 h-8 bg-white border border-slate-200 rounded shadow-sm flex items-center justify-center text-slate-500 hover:bg-slate-50 transition-colors">
      <span className="text-sm font-bold">{children}</span>
    </button>
  );
}

function MiniMetric({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <div className="bg-white/60 border border-brand-border rounded-lg px-2 py-2">
      <p className="text-[8px] text-slate-400 uppercase tracking-widest font-bold">{label}</p>
      <p className="text-[12px] text-slate-700 font-tech leading-none mt-1">
        {value}<span className="text-[8px] text-slate-400 ml-1">{unit}</span>
      </p>
    </div>
  );
}

function OverrideToggle({
  label,
  icon: Icon,
  active = false,
  disabled = false,
  pending = false,
  onClick,
}: {
  label: string;
  icon: LucideIcon;
  active?: boolean;
  disabled?: boolean;
  pending?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={cn(
        'glass p-5 flex flex-col items-center justify-center gap-4 bg-white hover:bg-slate-50 transition-colors cursor-pointer disabled:cursor-not-allowed disabled:opacity-60',
        active && 'ring-1 ring-slate-900/10',
      )}
      onClick={onClick}
      disabled={disabled}
    >
      <div className={cn(
        'p-3.5 rounded-lg transition-all duration-300',
        active ? 'bg-slate-900 text-white shadow-lg' : 'bg-slate-100 text-slate-400',
      )}>
        <Icon className="w-6 h-6" />
      </div>
      <div className="text-center">
        <h4 className="tech-font text-[9px] mb-1">{label}</h4>
        <span className={cn('text-[8px] font-tech tracking-tighter uppercase', active ? 'text-emerald-500' : 'text-slate-400')}>
          {pending ? 'SENDING' : active ? 'ACTIVE_MODE' : 'SEND_MODE'}
        </span>
      </div>
    </button>
  );
}

function MetricCircle({ label, value, max = 100, unit = '%', color }: { label: string; value: number; max?: number; unit?: string; color: Tone }) {
  const percentage = clampPercent((value / max) * 100);
  const circumference = 2 * Math.PI * 50;
  const offset = circumference - (percentage / 100) * circumference;
  const colorClass = color === 'emerald' ? 'text-emerald-500' : color === 'blue' ? 'text-blue-500' : color === 'amber' ? 'text-amber-500' : color === 'rose' ? 'text-rose-500' : color === 'slate' ? 'text-slate-500' : 'text-slate-500';

  return (
    <div className="flex flex-col items-center text-center">
      <div className="relative w-28 h-28 mb-3">
        <svg className="w-full h-full -rotate-90">
          <circle cx="56" cy="56" r="50" fill="none" stroke="#f1f5f9" strokeWidth="6" />
          <circle
            cx="56"
            cy="56"
            r="50"
            fill="none"
            stroke="currentColor"
            strokeWidth="6"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className={cn('transition-all duration-1000 ease-out', colorClass)}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-xl font-tech text-slate-800 leading-none">{formatNumber(value, 1)}</span>
          <span className="text-[8px] text-slate-400 font-tech mt-1">{unit}</span>
        </div>
      </div>
      <h4 className="tech-font text-slate-500 text-[9px]">{label}</h4>
    </div>
  );
}

function toSensorData(history: HistoryPoint[], key: SensorKey, fallback: number): SensorData[] {
  const source = history.length ? history.slice(-24) : [];
  if (!source.length) return [{ time: 0, value: fallback }];
  return source.map((point, index) => ({
    time: point.timestamp || index,
    value: Number.isFinite(Number(point[key])) ? Number(point[key]) : fallback,
  }));
}

function formatNumber(value: unknown, digits = 1) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '--';
  return numeric.toFixed(digits);
}

function formatTimestamp(value: string | null) {
  if (!value) return 'waiting';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}

function clampPercent(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

function getStateTone(label: string) {
  const normalised = String(label || '').toLowerCase();
  if (normalised === 'dangerous' || normalised === 'critical') {
    return {
      badge: 'text-rose-600 bg-rose-50 border-rose-100',
      panel: 'border-rose-200 bg-rose-50/20',
      text: 'text-rose-600',
      iconBg: 'bg-rose-500/10',
    };
  }
  if (normalised === 'warning') {
    return {
      badge: 'text-amber-600 bg-amber-50 border-amber-100',
      panel: 'border-amber-200 bg-amber-50/20',
      text: 'text-amber-600',
      iconBg: 'bg-amber-500/10',
    };
  }
  return {
    badge: 'text-blue-600 bg-blue-50 border-blue-100',
    panel: 'border-slate-200 bg-white/10',
    text: 'text-blue-600',
    iconBg: 'bg-blue-500/10',
  };
}
