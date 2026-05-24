import type { VercelRequest, VercelResponse } from "@vercel/node";
import { HfInference } from "@huggingface/inference";

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

  try {
    // convert base64 → Blob
    const imageBuffer = Buffer.from(image_b64, "base64");
    const imageBlob = new Blob([imageBuffer], { type: "image/jpeg" });

    const hf = new HfInference(token);
    const result = await hf.visualQuestionAnswering({
      model: "Salesforce/blip-vqa-base",
      inputs: { image: imageBlob, question: question.trim() },
    });

    return res.status(200).json({ answer: result.answer, question });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return res.status(502).json({ detail: message });
  }
}
