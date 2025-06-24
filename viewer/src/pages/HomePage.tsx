import { useState } from 'react';
import { useUploadAndRun } from '../hooks';
import UploadHero from '../components/UploadHero';
import ProcessingScreen from '../components/ProcessingScreen';
import ViewerPage from './ViewerPage';
import type { Label } from '../types/labels';

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const { status, labels = [], imageUrl, error } = useUploadAndRun(file);

  if (error) return <pre>{String(error)}</pre>;

  if (!file) {
    return <UploadHero onFileSelect={setFile} />;
  }

  if (status !== 'done' || !imageUrl) {
    return <ProcessingScreen />;
  }

  return (
    <ViewerPage
      src={imageUrl}
      labels={labels as Label[]}
    />
  );
}
