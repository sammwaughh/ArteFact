export interface Source {
  title: string;
  authors: string;
  year: number | string;
  doi?: string;
}

export interface Label {
  id: string;
  text: string;
  confidence: number;
  source: Source;
}

export interface PaintingMeta {
  title: string;
  artist: string;
  year: string;
  image: string; // path under /public/images
}

export interface Painting {
  painting: PaintingMeta;
  labels: Label[];
}
