import { createRoot } from 'react-dom/client';
import { ChakraProvider } from '@chakra-ui/react';
import App from './App';

const container = document.getElementById('root')!;
createRoot(container).render(
  <ChakraProvider>
    <App />
  </ChakraProvider>,
);
