import { Flex } from '@chakra-ui/react';
import PaintingViewer from '../components/PaintingViewer';
import Sidebar from '../components/Sidebar';
import type { Label } from '../types/labels';

interface Props { src: string; labels: Label[]; }

export default function ViewerPage({ src, labels }: Props) {
  return (
    <Flex h="calc(100vh - 56px)" overflow="hidden">
      <PaintingViewer src={src} alt="Uploaded painting" />
      <Sidebar labels={labels} />
    </Flex>
  );
}
