// Calls the Vercel serverless function at /api/predict (same origin).
// Falls back to a local FastAPI server when VITE_API_URL is set (local dev).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const API_BASE: string = (import.meta as any).env?.VITE_API_URL ?? "";

export interface PredictResult {
  answer: string;
  question: string;
}

export async function predict(
  file: File,
  question: string,
  onRetry?: () => void
): Promise<PredictResult> {
  const b64 = await fileToBase64(file);
  const body = JSON.stringify({ image_b64: b64, question });
  const headers = { "Content-Type": "application/json" };
  const url = `${API_BASE}/api/predict`;

  let res = await fetch(url, { method: "POST", headers, body });

  if (res.status === 503) {
    onRetry?.();
    await sleep(20_000);
    res = await fetch(url, { method: "POST", headers, body });
  }

  if (!res.ok) {
    const text = await res.text();
    let detail = text;
    try {
      detail = JSON.parse(text)?.detail ?? text;
    } catch { /* use raw text */ }
    throw new Error(detail || `Server error ${res.status}`);
  }

  return res.json() as Promise<PredictResult>;
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve((reader.result as string).split(",")[1]);
    reader.onerror = () => reject(reader.error);
  });
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
