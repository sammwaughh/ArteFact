/**
 * React hooks that orchestrate upload → run → poll.
 */

import { useEffect, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react'; // ← type-only import
import {
  requestPresign,
  createRun,
  getRun,
  type RunStatus, // ← type-only
} from './api';

interface UseUploadAndRunState {
  runId?: string;
  status?: RunStatus['status'];
  labels?: unknown[];
  imageUrl?: string;
  error?: Error;
}

/* ------------------------------------------------------------------ */
/** Polls the back-end every `intervalMs` until the run is done. */
function usePollRun(
  runId: string | undefined,
  setState: Dispatch<SetStateAction<UseUploadAndRunState>>,
  intervalMs = 3000,
) {
  useEffect(() => {
    if (!runId) return;

    const id = setInterval(async () => {
      try {
        const run = await getRun(runId);

        if (run.status === 'done' && run.outputKey) {
          clearInterval(id);

          // Fetch labels JSON from CloudFront
          const base = import.meta.env.VITE_CLOUDFRONT_URL;
          const labelsResp = await fetch(`${base}/${run.outputKey}`);
          const labels: unknown[] = await labelsResp.json();

          setState({
            runId,
            status: 'done',
            labels,
            imageUrl: `${base}/artifacts/${runId}.jpg`,
          });
        } else {
          // updater-function keeps TS happy
          setState((prev) => ({ ...prev, status: run.status }));
        }
      } catch (err: unknown) {
        setState({
          error: err instanceof Error ? err : new Error(String(err)),
        });
      }
    }, intervalMs);

    return () => clearInterval(id);
  }, [runId, intervalMs, setState]);
}

/* ------------------------------------------------------------------ */
/** Main orchestration hook: upload file → create run → poll status. */
export function useUploadAndRun(file: File | null): UseUploadAndRunState {
  const [state, setState] = useState<UseUploadAndRunState>({});

  // Step 1: kick off once we have a file
  useEffect(() => {
    if (!file) return;

    (async () => {
      try {
        const { runId, uploadUrl, s3Key } = await requestPresign(file.name);

        await fetch(uploadUrl, {
          method: 'PUT',
          headers: { 'Content-Type': file.type },
          body: file,
        });

        await createRun(runId, s3Key);
        setState({ runId, status: 'queued' });
      } catch (err: unknown) {
        setState({
          error: err instanceof Error ? err : new Error(String(err)),
        });
      }
    })();
  }, [file]);

  // Step 2: poll once runId exists
  usePollRun(state.runId, setState);

  return state;
}
