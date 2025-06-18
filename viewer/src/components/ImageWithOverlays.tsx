/* eslint-disable react/require-default-props */
export interface Label {
  label: string;
  score: number;
  evidence: unknown; // ← was any
}

interface Props {
  src: string;
  labels: Label[];
}

export default function ImageWithOverlays({ src /* , labels */ }: Props) {
  return <img src={src} alt="result" style={{ width: '100%' }} />;
}
