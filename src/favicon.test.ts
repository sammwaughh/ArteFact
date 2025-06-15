import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { resolve } from 'path';

describe('Favicon configuration', () => {
  it('should reference favicon-32.png in index.html', () => {
    const indexHtml = readFileSync(
      resolve(__dirname, '../index.html'),
      'utf-8',
    );

    // Check that favicon-32.png is referenced
    expect(indexHtml).toContain('favicon-32.png');

    // Check that the type is image/png
    expect(indexHtml).toContain('type="image/png"');

    // Check that old favicon.jpg is not referenced
    expect(indexHtml).not.toContain('favicon.jpg');
  });
});
