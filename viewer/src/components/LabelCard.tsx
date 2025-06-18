import {
  Box,
  Text,
  Badge,
  Stack,
  Link,
  useColorModeValue as mode,
} from '@chakra-ui/react';
import type { Label } from '../types/labels';

export default function LabelCard({ text, confidence, source }: Label) {
  return (
    <Box
      borderWidth="1px"
      borderRadius="md"
      p={3}
      bg={mode('white', 'gray.700')}
      shadow="sm"
    >
      <Stack spacing={2}>
        <Text fontSize="sm">{text}</Text>
        <Badge colorScheme="green">{Math.round(confidence * 100)} %</Badge>
        <Text fontSize="xs" color="gray.500">
          {source.doi ? (
            <Link href={`https://doi.org/${source.doi}`} isExternal>
              {source.title}
            </Link>
          ) : (
            source.title
          )}{' '}
          ({source.year}) — {source.authors}
        </Text>
      </Stack>
    </Box>
  );
}
