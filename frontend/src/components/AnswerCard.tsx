export interface QAPair {
  question: string;
  answer: string;
  timestamp: number;
}

interface Props {
  history: QAPair[];
}

export default function AnswerCard({ history }: Props) {
  if (history.length === 0) return null;

  return (
    <div className="space-y-3 animate-fade-in">
      <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-500">
        Answers
      </h3>
      {[...history].reverse().map((pair) => (
        <div
          key={pair.timestamp}
          className="glass rounded-2xl p-5 space-y-2 animate-slide-up"
        >
          <div className="flex items-start gap-3">
            <span className="mt-0.5 flex-shrink-0 w-6 h-6 rounded-full bg-purple-600/30 flex items-center justify-center">
              <svg className="w-3 h-3 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </span>
            <p className="text-slate-400 text-sm">{pair.question}</p>
          </div>
          <div className="flex items-start gap-3 ml-0">
            <span className="mt-0.5 flex-shrink-0 w-6 h-6 rounded-full bg-blue-600/30 flex items-center justify-center">
              <svg className="w-3 h-3 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </span>
            <p className="text-white font-medium text-base">{pair.answer}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
