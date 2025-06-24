import { Flex, Heading } from '@chakra-ui/react';

export default function AppHeader() {
  return (
    <Flex
      as="header"
      h="56px"
      px={4}
      align="center"
      borderBottomWidth="1px"
      bg="white"
      position="sticky"
      top={0}
      zIndex={10}
    >
      <Heading
        size="md"
        letterSpacing="tight"
        cursor="pointer"
        onClick={() => window.location.reload()}
      >
        ArteFact
      </Heading>
    </Flex>
  );
}