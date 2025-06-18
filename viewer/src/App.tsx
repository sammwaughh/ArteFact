import {
  ChakraProvider,
  Flex,
  Center,
  Spinner,
  Box,
  Image,
} from '@chakra-ui/react';
import { useEffect, useState } from 'react';
import type { Painting } from './types/labels';
import PaintingViewer from './components/PaintingViewer';
import Sidebar from './components/Sidebar';

export default function App() {
  const [data, setData] = useState<Painting | null>(null);
  const [error, setError] = useState<string | null>(null);

  // import.meta.env.BASE_URL is injected at build time and never changes
  const BASE = import.meta.env.BASE_URL;

  // ▶ set tab title
  useEffect(() => {
    if (data?.painting?.title) {
      document.title = `${data.painting.title} – Viewer`;
    } else {
      document.title = 'ArteFact Viewer';
    }
  }, [data]);

  useEffect(() => {
    fetch(`${BASE}data/nightwatch.labels.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(e.message));
  }, [BASE]); // ✅ satisfies react-hooks/exhaustive-deps

  return (
    <ChakraProvider>
      {/* Column layout: header at top, main area fills the rest */}
      <Flex direction="column" h="100vh">
        {/* --- HEADER --- */}
        <Box as="header" p={4} borderBottomWidth="1px">
          <Image
            src={`${BASE}images/logo-16-9.JPEG`}
            alt="ArteFact"
            h="80px"
            objectFit="contain"
          />
        </Box>

        {/* --- MAIN AREA --- */}
        {error ? (
          <Center flex="1" color="red.500">
            Error: {error}
          </Center>
        ) : !data ? (
          <Center flex="1">
            <Spinner size="xl" />
          </Center>
        ) : (
          <Flex flex="1">
            <PaintingViewer
              src={`${BASE}images/nightwatch.jpg`}
              alt={data.painting.title}
            />
            <Sidebar labels={data.labels} />
          </Flex>
        )}
      </Flex>
    </ChakraProvider>
  );
}
