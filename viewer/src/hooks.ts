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
        } else if (run.status === 'error') {
          clearInterval(id);
          setState({
            runId,
            status: 'error',
            error: new Error(run.errorMessage ?? 'Run failed on the server'),
          });
        } else {
          setState((prev) => ({ ...prev, status: run.status })); // queued / processing
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
        // NEW response shape: { runId, s3Key, upload:{ url, fields:{…} } }
        const { runId, s3Key, upload } = await requestPresign(file.name);

        // Build the <form> body required by S3's presigned POST
        const form = new FormData();
        Object.entries(upload.fields).forEach(([k, v]) =>
          form.append(k, v as string),
        );
        form.append('Content-Type', file.type); // must satisfy the policy
        form.append('file', file);              // ✱ must be the last field ✱

        await fetch(upload.url, {
          method: 'POST',
          body: form,
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
