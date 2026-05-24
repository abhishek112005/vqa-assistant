import type { VercelRequest, VercelResponse } from "@vercel/node";

const HF_MODEL = "Salesforce/blip-vqa-base";
const HF_API_URL = `https://api-inference.huggingface.co/models/${HF_MODEL}`;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({ detail: "Method not allowed" });
  }

  const { image_b64, question } = req.body as {
    image_b64?: string;
    question?: string;
  };

  if (!question?.trim()) {
    return res.status(422).json({ detail: "Question must not be empty." });
  }
  if (!image_b64) {
    return res.status(422).json({ detail: "image_b64 is required." });
  }

  const token = process.env.HF_TOKEN;
  if (!token) {
    return res.status(503).json({ detail: "HF_TOKEN not configured on server." });
  }

  let hfRes: Response;
  try {
    hfRes = await fetch(HF_API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ inputs: { image: image_b64, question } }),
    });
  } catch (err: unknown) {
    return res.status(502).json({ detail: `HuggingFace unreachable: ${err}` });
  }

  if (hfRes.status === 503) {
    return res.status(503).json({ detail: "Model is loading on HuggingFace, retry in ~20 s." });
  }

  const text = await hfRes.text();
  if (!hfRes.ok) {
    return res.status(hfRes.status).json({ detail: text });
  }

  let answer = "";
  try {
    const json = JSON.parse(text);
    if (Array.isArray(json) && json.length > 0) answer = json[0].answer ?? "";
    else if (typeof json === "object") answer = json.answer ?? "";
    else answer = String(json);
  } catch {
    answer = text;
  }

  return res.status(200).json({ answer, question });
}
