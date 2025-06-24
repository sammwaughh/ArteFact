import { Flex, Image, Heading } from '@chakra-ui/react';

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
      <Image
        src={`${import.meta.env.BASE_URL}images/logo-16-9.JPEG`}
        alt="ArteFact logo"
        h="36px"
        mr={3}
      />
      <Heading size="md" letterSpacing="tight">
        ArteFact
      </Heading>
    </Flex>
  );
}