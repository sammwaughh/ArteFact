import { Box, Heading, Stack } from '@chakra-ui/react';
import type { Label } from '../types/labels';
import LabelCard from './LabelCard';

interface Props {
  labels: Label[];
}

export default function Sidebar({ labels }: Props) {
  return (
    <Box w="320px" p={4} borderLeftWidth="1px" overflowY="auto" bg="gray.100">
      <Heading as="h2" size="sm" mb={4}>
        Labels ({labels.length})
      </Heading>

      <Stack spacing={3}>
        {labels.map((l) => (
          <LabelCard key={l.id} {...l} />
        ))}
      </Stack>
    </Box>
  );
}
