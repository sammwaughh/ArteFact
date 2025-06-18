import { useState } from 'react';
import { useUploadAndRun } from '../hooks';
import ImageWithOverlays, { Label } from '../components/ImageWithOverlays';

/* Helper component declared first to avoid use-before-define rule */
function ViewerInner({
  labels,
  imageUrl,
}: {
  labels: Label[];
  imageUrl: string;
}) {
  return (
    <div style={{ marginTop: '2rem' }}>
      <ImageWithOverlays src={imageUrl} labels={labels} />
    </div>
  );
}

/* ------------------------------------------------------------------ */

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const { status, labels, imageUrl, error } = useUploadAndRun(file);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFile(e.target.files?.[0] ?? null);
  };

  return (
    <main style={{ padding: '2rem' }}>
      <h1>Upload an image</h1>

      <input type="file" accept="image/*" onChange={handleFileSelect} />

      {status === 'queued' && <p>Processing… ⏳</p>}

      {status === 'done' && labels && imageUrl && (
        <ViewerInner labels={labels as Label[]} imageUrl={imageUrl} />
      )}

      {error && (
        <pre style={{ color: 'red', whiteSpace: 'pre-wrap' }}>
          {String(error)}
        </pre>
      )}
    </main>
  );
}
