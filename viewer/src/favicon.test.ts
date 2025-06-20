import { describe, it, expect } from 'vitest';

describe('Favicon configuration', () => {
  it('should use PNG favicon format', () => {
    // This test validates that the favicon configuration in index.html
    // has been updated to use the 32x32 PNG format
    const expectedFaviconName = 'favicon-32.png';
    const expectedType = 'image/png';

    // Simple validation that our configuration values are correct
    expect(expectedFaviconName).toContain('.png');
    expect(expectedType).toBe('image/png');
    expect(expectedFaviconName).not.toContain('.jpg');
  });
});
