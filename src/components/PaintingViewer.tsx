import { Image, Box } from '@chakra-ui/react';

interface Props {
  src: string;
  alt: string;
}

/**
 * Full-height, full-width painting pane.
 * Scales the image to fit while preserving aspect ratio.
 */
export default function PaintingViewer({ src, alt }: Props) {
  return (
    <Box
      flex="1"
      bg="gray.50"
      display="flex"
      alignItems="center"
      justifyContent="center"
    >
      <Image
        src={src}
        alt={alt}
        maxH="100%"
        maxW="100%"
        objectFit="contain"
        fallbackSrc={`${import.meta.env.BASE_URL}vite.svg`}
      />
    </Box>
  );
}
