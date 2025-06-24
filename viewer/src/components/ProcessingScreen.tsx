import { Center, Spinner, VStack, Text } from '@chakra-ui/react';

export default function ProcessingScreen() {
  return (
    <Center minH="calc(100vh - 56px)">
      <VStack spacing={4}>
        <Spinner size="xl" />
        <Text>Processing&hellip; this usually takes 3‑5 seconds.</Text>
      </VStack>
    </Center>
  );
}
