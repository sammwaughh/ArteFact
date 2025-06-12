import { ChakraProvider, Flex, Center, Spinner } from '@chakra-ui/react';
import { useEffect, useState } from 'react';
import type { Painting } from './types/labels';
import PaintingViewer from './components/PaintingViewer';
import Sidebar from './components/Sidebar';

export default function App() {
  const [data, setData] = useState<Painting | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/data/nightwatch.labels.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  return (
    <ChakraProvider>
      {error ? (
        <Center h="100vh" color="red.500">
          Error: {error}
        </Center>
      ) : !data ? (
        <Center h="100vh">
          <Spinner size="xl" />
        </Center>
      ) : (
        <Flex h="100vh">
          <PaintingViewer
            src="/images/nightwatch.jpg"
            alt={data.painting.title}
          />
          <Sidebar labels={data.labels} />
        </Flex>
      )}
    </ChakraProvider>
  );
}
