// eslint-disable-next-line @typescript-eslint/no-explicit-any
const API_BASE: string = (import.meta as any).env?.VITE_API_URL ?? "http://localhost:7860";

export interface PredictResult {
  answer: string;
  question: string;
}

// Retry once when HF model is still loading (503)
export async function predict(
  file: File,
  question: string,
  onRetry?: () => void
): Promise<PredictResult> {
  const b64 = await fileToBase64(file);
  const body = JSON.stringify({ image_b64: b64, question });
  const headers = { "Content-Type": "application/json" };

  let res = await fetch(`${API_BASE}/predict`, { method: "POST", headers, body });

  if (res.status === 503) {
    onRetry?.();
    await sleep(20_000);
    res = await fetch(`${API_BASE}/predict`, { method: "POST", headers, body });
  }

  if (!res.ok) {
    // Read the body once as text, then try to parse as JSON for the detail field
    const text = await res.text();
    let detail = text;
    try {
      const json = JSON.parse(text);
      detail = json?.detail ?? text;
    } catch {
      // text is not JSON — use as-is
    }
    throw new Error(detail || `Server error ${res.status}`);
  }

  return res.json() as Promise<PredictResult>;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    return res.ok;
  } catch {
    return false;
  }
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      const result = reader.result as string;
      resolve(result.split(",")[1]);
    };
    reader.onerror = () => reject(reader.error);
  });
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
