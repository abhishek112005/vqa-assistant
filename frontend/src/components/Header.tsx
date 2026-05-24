export default function Header() {
  return (
    <header className="text-center py-12 px-4">
      <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full glass text-xs text-purple-300 mb-6 tracking-widest uppercase">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        Powered by BLIP · Open-ended VQA
      </div>
      <h1 className="text-5xl sm:text-6xl font-bold mb-4">
        <span className="gradient-text">Visual Question</span>
        <br />
        <span className="text-slate-100">Answering</span>
      </h1>
      <p className="text-slate-400 max-w-md mx-auto text-lg leading-relaxed">
        Upload any image and ask a question in plain English.
        <br />
        BLIP reads the image and generates the answer.
      </p>
    </header>
  );
}
