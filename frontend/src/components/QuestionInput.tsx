import { useState, KeyboardEvent } from "react";

const EXAMPLES = [
  "What is in the image?",
  "What color is the main object?",
  "How many people are there?",
  "Is it daytime or nighttime?",
  "What is the person doing?",
  "What animals are present?",
];

interface Props {
  onSubmit: (q: string) => void;
  loading: boolean;
  loadingLabel?: string;
  disabled: boolean;
}

export default function QuestionInput({ onSubmit, loading, loadingLabel, disabled }: Props) {
  const [question, setQuestion] = useState("");

  const submit = () => {
    const q = question.trim();
    if (!q || loading || disabled) return;
    onSubmit(q);
    setQuestion("");
  };

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") submit();
  };

  return (
    <div className="space-y-4">
      {/* Example chips */}
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            onClick={() => setQuestion(ex)}
            className="text-xs px-3 py-1.5 rounded-full glass text-slate-300 hover:text-purple-300 hover:border-purple-500/40 transition-all"
          >
            {ex}
          </button>
        ))}
      </div>

      {/* Input row */}
      <div className="flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={onKey}
          placeholder="Ask anything about the image…"
          disabled={loading}
          className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-slate-100 placeholder-slate-500
            focus:outline-none focus:border-purple-500/60 focus:ring-1 focus:ring-purple-500/30
            disabled:opacity-50 transition-all text-sm"
        />
        <button
          onClick={submit}
          disabled={!question.trim() || loading || disabled}
          className="px-5 py-3 rounded-xl font-medium text-sm transition-all
            bg-gradient-to-r from-purple-600 to-blue-600
            hover:from-purple-500 hover:to-blue-500
            disabled:opacity-40 disabled:cursor-not-allowed
            flex items-center gap-2 whitespace-nowrap"
        >
          {loading ? (
            <>
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              {loadingLabel ?? "Thinking…"}
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Ask
            </>
          )}
        </button>
      </div>
    </div>
  );
}
