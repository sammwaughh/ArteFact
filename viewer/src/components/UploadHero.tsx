import {
  Center, VStack, Icon, Button, Text, Input,
} from '@chakra-ui/react';
import { FiUploadCloud } from 'react-icons/fi';
import { useRef } from 'react';

interface Props { onFileSelect: (f: File) => void; }

export default function UploadHero({ onFileSelect }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <Center minH="calc(100vh - 56px)">
      <VStack spacing={6}>
        <Icon as={FiUploadCloud} boxSize={16} color="gray.400" />
        <Button
          colorScheme="teal"
          size="lg"
          onClick={() => inputRef.current?.click()}
        >
          Select an image
        </Button>
        <Text fontSize="sm" color="gray.500">
          JPEG / PNG, up to 50 MB
        </Text>
        <Input
          type="file"
          accept="image/*"
          ref={inputRef}
          display="none"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFileSelect(f);
          }}
        />
      </VStack>
    </Center>
  );
}
