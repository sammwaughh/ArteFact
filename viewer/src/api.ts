/**
 * Thin fetch wrapper around the Flask backend.
 */

export interface PresignResponse {
  runId: string;
  uploadUrl: string;
  s3Key: string;
}

export interface RunStatus {
  runId: string;
  status: 'queued' | 'processing' | 'done' | 'error';
  outputKey?: string;
  errorMessage?: string; // ← new
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/* ----------------------------- endpoints ---------------------------- */

export async function requestPresign(
  fileName: string,
): Promise<PresignResponse> {
  const resp = await fetch(`${API_URL}/presign`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fileName }),
  });
  if (!resp.ok) throw new Error('presign failed');
  return resp.json();
}

export async function createRun(runId: string, s3Key: string) {
  const resp = await fetch(`${API_URL}/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ runId, s3Key }),
  });
  if (!resp.ok) throw new Error('createRun failed');
}

export async function getRun(runId: string): Promise<RunStatus> {
  const resp = await fetch(`${API_URL}/runs/${runId}`);
  if (!resp.ok) throw new Error('getRun failed');
  return resp.json();
}
