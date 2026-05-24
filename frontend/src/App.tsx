import { useCallback, useState } from "react";
import Header from "./components/Header";
import ImageUpload from "./components/ImageUpload";
import QuestionInput from "./components/QuestionInput";
import AnswerCard, { QAPair } from "./components/AnswerCard";
import ErrorBanner from "./components/ErrorBanner";
import { predict } from "./api";

type Status = "idle" | "loading" | "warming" | "ready" | "error";

const STATUS_LABEL: Record<string, string> = {
  loading: "Thinking…",
  warming: "Model warming up, retrying in ~20 s…",
};

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [history, setHistory] = useState<QAPair[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setHistory([]);
    setErrorMsg(null);
  }, []);

  const handleQuestion = async (question: string) => {
    if (!file) return;
    setStatus("loading");
    setErrorMsg(null);
    try {
      const result = await predict(file, question, () => setStatus("warming"));
      setHistory((prev: QAPair[]) => [
        ...prev,
        { question: result.question, answer: result.answer, timestamp: Date.now() },
      ]);
      setStatus("ready");
    } catch (e: unknown) {
      setStatus("error");
      setErrorMsg(e instanceof Error ? e.message : "Something went wrong.");
    }
  };

  const isLoading = status === "loading" || status === "warming";

  return (
    <div className="min-h-screen bg-[#0a0a14]">
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 -right-40 w-96 h-96 rounded-full bg-purple-600/10 blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-96 h-96 rounded-full bg-blue-600/10 blur-3xl" />
      </div>

      <div className="relative z-10 max-w-5xl mx-auto px-4 pb-20">
        <Header />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="glass rounded-2xl p-4">
              <h2 className="text-xs uppercase tracking-widest text-slate-500 font-semibold mb-3">
                Image
              </h2>
              <ImageUpload onFile={handleFile} previewUrl={previewUrl} />
            </div>
          </div>

          <div className="space-y-5">
            <div className="glass rounded-2xl p-5">
              <h2 className="text-xs uppercase tracking-widest text-slate-500 font-semibold mb-4">
                Question
              </h2>
              <QuestionInput
                onSubmit={handleQuestion}
                loading={isLoading}
                loadingLabel={STATUS_LABEL[status]}
                disabled={!file}
              />
              {!file && (
                <p className="mt-3 text-xs text-slate-600 text-center">
                  Upload an image to enable the question box
                </p>
              )}
            </div>

            {errorMsg && (
              <ErrorBanner
                message={errorMsg}
                onDismiss={() => setErrorMsg(null)}
              />
            )}

            <AnswerCard history={history} />
          </div>
        </div>

        <p className="text-center text-slate-600 text-xs mt-16">
          Built with BLIP · FastAPI · React · Render · Vercel
        </p>
      </div>
    </div>
  );
}
