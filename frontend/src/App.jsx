import { useState, useRef, useCallback } from "react";

const API_URL = import.meta.env.VITE_API_URL || "https://your-space.hf.space";

const CLASS_INFO = {
  glioma: {
    solid: "bg-red-500",
    text: "text-red-500",
    bg: "bg-red-50",
    border: "border-red-200",
    icon: "⚠",
    desc: "A tumor originating in the glial cells of the brain. Requires immediate medical attention.",
  },
  meningioma: {
    solid: "bg-orange-500",
    text: "text-orange-500",
    bg: "bg-orange-50",
    border: "border-orange-200",
    icon: "◉",
    desc: "A tumor arising from the meninges — the membranes surrounding the brain. Often benign.",
  },
  notumor: {
    solid: "bg-green-500",
    text: "text-green-500",
    bg: "bg-green-50",
    border: "border-green-200",
    icon: "✓",
    desc: "No tumor detected. The scan appears healthy.",
  },
  pituitary: {
    solid: "bg-violet-500",
    text: "text-violet-500",
    bg: "bg-purple-50",
    border: "border-violet-200",
    icon: "◎",
    desc: "A tumor located in the pituitary gland at the base of the brain. Often treatable.",
  },
};

function ConfidenceBar({ label, value, colorClass }) {
  return (
    <div className="mb-2.5">
      <div className="flex justify-between mb-1">
        <span className="text-[13px] font-mono text-slate-500 uppercase tracking-[0.05em]">
          {label}
        </span>
        <span className="text-[13px] font-mono text-slate-800 font-semibold">
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${colorClass} transition-[width] duration-1000 ease-in-out`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  );
}

export default function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef();

  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) {
      setError("Please upload a JPEG or PNG image.");
      return;
    }
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const handleAnalyze = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", image);
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to connect to the API.");
    } finally {
      setLoading(false);
    }
  };

  const info = result ? CLASS_INFO[result.predicted_class] : null;

  return (
    <div className="min-h-screen bg-slate-50 font-sans">
      {/* Header */}
      <header className="bg-slate-900 px-10 py-3 flex items-center justify-between border-b border-slate-800">
        <div className="flex items-center gap-3">
          
          <div>
            <div className="text-slate-100 text-[17px] font-bold tracking-widest">
              NeuraScan
            </div>
            <div className="text-white text-[11px] tracking-[0.08em] uppercase">
              Brain Tumor MRI Classifier
            </div>
          </div>
        </div>
        <div className="text-[11px] text-slate-600 font-mono bg-slate-800 px-3 py-1.5 rounded-md border border-slate-700">
          EfficientNetB0 · ViT-Base · 94.3% accuracy
        </div>
      </header>

      <main className="max-w-[1200px] mx-auto px-6 py-12">
        {/* Hero */}
        <div className="text-center mb-12">
          <h1 className="text-[clamp(28px,5vw,44px)] font-extrabold text-slate-900  leading-[1.1] mb-4">
            MRI Analysis in{" "}
            <span className="bg-gradient-to-br from-teal-500 to-emerald-500 bg-clip-text text-transparent">
              seconds
            </span>
          </h1>
          <p className="text-slate-500 text-[17px]  mx-auto leading-relaxed">
            Upload a brain MRI scan. Our fine-tuned Vision Transformer
            classifies it into 4 categories instantly.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-6 items-stretch">
          {/* Upload panel */}
          <div className="flex flex-col h-full">
            <div
              onClick={() => fileRef.current.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              className={`flex-1 flex flex-col items-center justify-center border-2 border-dashed rounded-2xl px-6 py-6 text-center cursor-pointer transition-colors duration-200 mb-4 ${
                dragging ? "border-teal-500 bg-indigo-50" : "border-slate-300 bg-white"
              }`}
            >
              {preview ? (
                <img
                  src={preview}
                  alt="MRI preview"
                  className="max-h-[280px] max-w-full rounded-[10px] object-contain"
                />
              ) : (
                <>
                  <div className="text-4xl mb-3">🔬</div>
                  <div className="text-slate-800 font-semibold mb-1.5">
                    Drop your MRI scan here
                  </div>
                  <div className="text-slate-400 text-sm">
                    or click to browse · JPEG, PNG
                  </div>
                </>
              )}
            </div>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFile(e.target.files[0])}
            />

            <button
              onClick={handleAnalyze}
              disabled={!image || loading}
              className={`w-full py-3.5 rounded-xl text-[15px] font-bold tracking-[-0.01em] transition-all duration-200 ${
                !image || loading
                  ? "bg-slate-200 text-slate-400 cursor-not-allowed"
                  : "bg-gradient-to-br from-teal-500 to-emerald-500 text-white cursor-pointer shadow-[0_4px_20px_rgba(99,102,241,0.35)]"
              }`}
            >
              {loading ? "Analyzing…" : "Analyze MRI"}
            </button>

            {error && (
              <div className="mt-3 px-4 py-3 bg-red-50 border border-red-200 rounded-[10px] text-red-600 text-sm">
                {error}
              </div>
            )}
          </div>

          {/* results and info */}
          <div>
            {result && info ? (
              <div className="animate-fade-in">
                {/* Prediction card */}
                <div className={`${info.bg} border-[1.5px] ${info.border} rounded-2xl p-5 mb-4`}>
                  <div className="flex items-center gap-3 mb-3">
                    <div className={`w-11 h-11 rounded-xl ${info.solid} text-white flex items-center justify-center text-xl font-bold shrink-0`}>
                      {info.icon}
                    </div>
                    <div>
                      <div className="text-[11px] text-slate-500 uppercase tracking-[0.08em] font-mono">
                        Prediction
                      </div>
                      <div className={`text-[22px] font-extrabold ${info.text} tracking-tight capitalize`}>
                        {result.predicted_class}
                      </div>
                    </div>
                    <div className="ml-auto text-right">
                      <div className="text-[11px] text-slate-500 uppercase tracking-[0.08em] font-mono">
                        Confidence
                      </div>
                      <div className={`text-[26px] font-extrabold ${info.text} tracking-tight`}>
                        {(result.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  <p className="text-slate-600 text-sm leading-relaxed m-0">
                    {info.desc}
                  </p>
                </div>

                {/* Score breakdown */}
                <div className="bg-white border-[1.5px] border-slate-200 rounded-2xl px-5 py-4">
                  <div className="text-[11px] text-slate-400 uppercase tracking-[0.08em] font-mono mb-4">
                    Score Breakdown
                  </div>
                  {Object.entries(result.all_scores)
                    .sort((a, b) => b[1] - a[1])
                    .map(([cls, score]) => (
                      <ConfidenceBar
                        key={cls}
                        label={cls}
                        value={score}
                        colorClass={CLASS_INFO[cls].solid}
                      />
                    ))}
                </div>

                {result.low_confidence && (
                  <div className="mt-3 px-4 py-3 bg-amber-50 border border-amber-200 rounded-[10px] text-amber-800 text-[13px]">
                    ⚠ Low confidence prediction. Consider consulting a specialist.
                  </div>
                )}

                {/* Disclaimer */}
                <div className="mt-3 px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-lg text-slate-400 text-xs leading-relaxed">
                  🔬 Research tool only. Not a substitute for professional medical diagnosis.
                </div>
              </div>
            ) : (
              <div>
                <div className="text-slate-400 text-xs uppercase tracking-[0.1em] font-mono mb-4">
                  Detectable conditions
                </div>
                <div className="grid grid-cols-1 gap-3">
                  {Object.entries(CLASS_INFO).map(([cls, info]) => (
                    <div key={cls} className={`bg-white border-[1.5px] ${info.border} rounded-xl p-4`}>
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`text-base font-bold ${info.text}`}>{info.icon}</span>
                        <span className="font-bold text-slate-900 capitalize text-sm">{cls}</span>
                      </div>
                      <p className="text-slate-500 text-xs leading-relaxed m-0">{info.desc}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}